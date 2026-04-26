#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import set_seed

from ctx_to_lora.data.definitions import CTX_AFFIXES, MULTI_ANSWER_DATASETS
from ctx_to_lora.data.processing import get_tokenized_dataset, load_answers
from ctx_to_lora.device import get_autocast_context, get_default_device, should_use_flash_attn
from ctx_to_lora.eval_utils import compute_qa_f1_score, normalize_answer
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from ctx_to_lora.utils import clear_gpu, get_run_name


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Doc-to-LoRA robustness under a queue of accumulated adapters."
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--base_model_path")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--eval_batch_size_gen", type=int, default=1)
    parser.add_argument("--max_test_samples_per_ds", type=int, default=50)
    parser.add_argument("--max_ctx_chunk_len", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--queue_lengths", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--history_scale", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_iterative_mode", action="store_true")
    parser.add_argument("--output_dir")
    return parser.parse_args()


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "queue_eval.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def canonical_tokenizer_name(name_or_path: str) -> str:
    if name_or_path in CTX_AFFIXES:
        return name_or_path
    lower_name = str(name_or_path).lower()
    if "gemma-2-2b-it" in lower_name:
        return "google/gemma-2-2b-it"
    if "qwen3-4b-instruct-2507" in lower_name:
        return "Qwen/Qwen3-4B-Instruct-2507"
    if "mistral-7b-instruct-v0.2" in lower_name:
        return "mistralai/Mistral-7B-Instruct-v0.2"
    raise KeyError(f"Unknown tokenizer name for CTX_AFFIXES: {name_or_path}")


def build_output_dir(args) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_path).resolve().parent.parent
    mode = "iterative" if args.use_iterative_mode else "batch"
    run_name = get_run_name(seed_str=time.strftime("%Y%m%d-%H%M%S"))
    return checkpoint_dir / f"queue-results-{mode}" / run_name


def clone_raw_loras(raw_loras: dict[str, dict[str, torch.Tensor]], device):
    out = {}
    for module_name, module_loras in raw_loras.items():
        out[module_name] = {
            "A": module_loras["A"].to(device),
            "B": module_loras["B"].to(device),
        }
    return out


def scale_raw_lora(
    raw_loras: dict[str, dict[str, torch.Tensor]],
    scale: float,
    device,
) -> dict[str, dict[str, torch.Tensor]]:
    out = {}
    for module_name, module_loras in raw_loras.items():
        out[module_name] = {
            "A": module_loras["A"].to(device) * scale,
            "B": module_loras["B"].to(device),
        }
    return out


def concat_raw_loras(
    entries: list[tuple[dict[str, dict[str, torch.Tensor]], int, float]],
    device,
) -> tuple[dict[str, dict[str, torch.Tensor]], int]:
    first_raw_lora = entries[0][0]
    combined = {}
    total_chunks = sum(n_chunks for _, n_chunks, _ in entries)
    for module_name in first_raw_lora:
        all_a = []
        all_b = []
        for raw_lora, n_chunks, scale in entries:
            scaled = scale_raw_lora(raw_lora, scale, device)
            all_a.append(scaled[module_name]["A"])
            all_b.append(scaled[module_name]["B"])
        combined[module_name] = {
            "A": torch.cat(all_a, dim=0),
            "B": torch.cat(all_b, dim=0),
        }
    return combined, total_chunks


def prepare_ctx_tensors(ctx_ids) -> tuple[torch.Tensor, torch.Tensor, int]:
    if isinstance(ctx_ids, torch.Tensor):
        if ctx_ids.ndim == 1:
            chunks = [ctx_ids.long()]
        else:
            chunks = [row.long() for row in ctx_ids]
    else:
        chunks = [torch.as_tensor(x, dtype=torch.long) for x in ctx_ids]
    attn_masks = [torch.ones_like(chunk) for chunk in chunks]
    padded_ctx_ids = pad_sequence(chunks, batch_first=True, padding_value=0)
    padded_attn_mask = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    return padded_ctx_ids, padded_attn_mask, len(chunks)


def decode_label(sample, tokenizer) -> str:
    labels = torch.as_tensor(sample["labels"])
    start_idx = int(torch.argmax((labels != -100).to(torch.int64)).item())
    label_toks = labels[start_idx:]
    label_toks = torch.where(
        label_toks == -100,
        torch.tensor(tokenizer.pad_token_id, dtype=label_toks.dtype),
        label_toks,
    )
    return tokenizer.decode(label_toks.tolist(), skip_special_tokens=True).strip()


def prompt_input_ids(sample) -> torch.Tensor:
    input_ids = torch.as_tensor(sample["input_ids"]).long()
    labels = torch.as_tensor(sample["labels"])
    idx = int(torch.argmax((labels != -100).to(torch.int64)).item())
    idx = max(1, idx)
    return input_ids[:idx]


def extract_generated_answer(tokenizer, generated_ids: torch.Tensor, input_len: int) -> str:
    gen_toks = generated_ids[input_len:].detach().cpu().numpy()
    tokenizer_name = canonical_tokenizer_name(tokenizer.name_or_path)
    suffix = np.array(CTX_AFFIXES[tokenizer_name]["suffix"])
    for i in range(len(gen_toks) - len(suffix), -1, -1):
        if all(gen_toks[i : i + len(suffix)] == suffix):
            gen_toks = gen_toks[i + len(suffix) :]
            break
    return tokenizer.decode(gen_toks.tolist(), skip_special_tokens=True).strip()


def load_model_and_tokenizers(args, device):
    state_dict = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    ctx_encoder_args = state_dict["ctx_encoder_args"]
    resolved_base_model_name = args.base_model_path or state_dict["base_model_name_or_path"]
    if args.base_model_path:
        state_dict["base_model_name_or_path"] = args.base_model_path
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict,
        train=False,
        use_flash_attn=should_use_flash_attn(True, device),
        use_sequence_packing=False,
    )
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    model.enable_iterative_mode(args.use_iterative_mode)

    tokenizer = get_tokenizer(resolved_base_model_name, train=False)

    ctx_model_name = ctx_encoder_args.ctx_encoder_model_name_or_path
    if ctx_model_name is None:
        ctx_model_name = resolved_base_model_name
    ctx_tokenizer = get_tokenizer(ctx_model_name, train=False)
    return model, tokenizer, ctx_tokenizer, {
        "base_model_name_or_path": resolved_base_model_name,
        "ctx_encoder_args": ctx_encoder_args,
    }


def load_dataset_for_eval(args, model, tokenizer, ctx_tokenizer, ds_name):
    base_model_max_len = getattr(model.base_model.config, "max_position_embeddings", 8192)
    ctx_model_max_len = getattr(
        model.ctx_encoder.base_model.config, "max_position_embeddings", None
    )
    ds = get_tokenized_dataset(
        ds_name=ds_name,
        split=args.split,
        max_qas_len=-1,
        max_qas_per_sample=1,
        base_model_max_len=base_model_max_len,
        tokenizer=tokenizer,
        ctx_model_max_len=ctx_model_max_len,
        ctx_tokenizer=ctx_tokenizer,
        max_ctx_chunk_len=args.max_ctx_chunk_len,
        min_ctx_chunk_len=-1,
        num_chunk_probs=None,
        max_ctx_chunk_num=None,
        add_ctx_to_chat=False,
        add_negative_prompt=False,
        use_kl_loss=False,
        max_new_tokens=args.max_new_tokens,
        add_self_distill_template=False,
        set_format="pt",
        truncate_if_too_long_inp=False,
        truncate_if_too_long_ctx=False,
        flip_ctx_inp=False,
    )
    selected_indices = np.random.permutation(len(ds))[: args.max_test_samples_per_ds]
    ds = ds.select(selected_indices)

    if ds_name in MULTI_ANSWER_DATASETS:
        answers_ds = load_answers(ds_name, args.split).select(selected_indices)
        answers = list(answers_ds["answers"])
    else:
        answers = [None] * len(ds)
    return ds, answers


def precompute_sample_adapters(model, dataset, tokenizer, device):
    logger.info("Precomputing raw LoRA adapters for %d samples", len(dataset))
    cached = []
    for sample_idx, sample in enumerate(dataset):
        model.reset()
        ctx_ids, ctx_attn_mask, n_chunks = prepare_ctx_tensors(sample["ctx_ids"])
        ctx_ids = ctx_ids.to(device)
        ctx_attn_mask = ctx_attn_mask.to(device)
        with torch.inference_mode(), get_autocast_context(device):
            raw_loras, _ = model.generate_weights(ctx_ids, ctx_attn_mask)

        cached_raw_loras = {
            module_name: {
                "A": module_loras["A"].detach().cpu(),
                "B": module_loras["B"].detach().cpu(),
            }
            for module_name, module_loras in raw_loras.items()
        }
        cached.append(
            {
                "raw_loras": cached_raw_loras,
                "n_chunks": n_chunks,
                "input_ids": prompt_input_ids(sample),
                "label": decode_label(sample, tokenizer),
                "ctx_ids_len": int(sample["ctx_ids_len"]),
            }
        )
        if (sample_idx + 1) % 10 == 0 or sample_idx + 1 == len(dataset):
            logger.info("Prepared %d/%d adapters", sample_idx + 1, len(dataset))
    clear_gpu()
    return cached


def answers_for_dataset(ds_name, cached_samples, answers):
    if ds_name in MULTI_ANSWER_DATASETS:
        return answers
    return [[sample["label"]] for sample in cached_samples]


def evaluate_queue_length(
    model,
    tokenizer,
    cached_samples,
    answers_list,
    queue_length,
    history_scale,
    max_new_tokens,
    device,
):
    pred_texts = []
    per_sample_rows = []

    start_time = time.perf_counter()
    for sample_idx, sample in enumerate(cached_samples):
        left_idx = max(0, sample_idx - queue_length + 1)
        queue_entries = []
        for idx in range(left_idx, sample_idx + 1):
            scale = 1.0 if idx == sample_idx else history_scale
            queue_entries.append(
                (cached_samples[idx]["raw_loras"], cached_samples[idx]["n_chunks"], scale)
            )

        raw_loras, total_chunks = concat_raw_loras(queue_entries, device)

        model.reset()
        model.patch_lora_forward()
        model.generated_loras = raw_loras

        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode(), get_autocast_context(device):
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                n_ctx_chunks=torch.tensor([total_chunks], dtype=torch.int32, device=device),
                max_new_tokens=max_new_tokens,
            )

        generated_text = extract_generated_answer(tokenizer, outputs[0], input_ids.shape[1])
        pred_texts.append(generated_text)
        per_sample_rows.append(
            {
                "sample_idx": sample_idx,
                "queue_length": queue_length,
                "label": sample["label"],
                "generated": generated_text,
                "ctx_ids_len": sample["ctx_ids_len"],
                "history_count": len(queue_entries) - 1,
            }
        )

    runtime = time.perf_counter() - start_time
    metric_values, per_sample_metric = compute_qa_f1_score(pred_texts, answers_list)
    metrics = {
        **metric_values,
        "runtime": runtime,
        "samples_per_second": len(cached_samples) / runtime if runtime > 0 else 0.0,
        "steps_per_second": len(cached_samples) / runtime if runtime > 0 else 0.0,
        "num_samples": len(cached_samples),
    }
    for row, f1, precision, recall in zip(
        per_sample_rows,
        per_sample_metric["qa_f1_score"],
        per_sample_metric["qa_precision"],
        per_sample_metric["qa_recall"],
    ):
        row["qa_f1_score"] = f1
        row["qa_precision"] = precision
        row["qa_recall"] = recall

    clear_gpu()
    return metrics, per_sample_rows


def write_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_summary_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    output_dir = build_output_dir(args)
    setup_logging(output_dir)

    set_seed(args.seed)
    np.random.seed(args.seed)
    device = get_default_device()
    logger.info("Running queue evaluation on device: %s", device)
    logger.info("Output dir: %s", output_dir)

    model, tokenizer, ctx_tokenizer, state_dict = load_model_and_tokenizers(
        args, device
    )

    summary_rows = []
    per_sample_rows = []

    run_metadata = {
        "checkpoint_path": args.checkpoint_path,
        "base_model_path": args.base_model_path,
        "datasets": args.datasets,
        "split": args.split,
        "max_test_samples_per_ds": args.max_test_samples_per_ds,
        "max_ctx_chunk_len": args.max_ctx_chunk_len,
        "max_new_tokens": args.max_new_tokens,
        "queue_lengths": args.queue_lengths,
        "history_scale": args.history_scale,
        "seed": args.seed,
        "use_iterative_mode": args.use_iterative_mode,
        "mode": "iterative" if args.use_iterative_mode else "batch",
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "ctx_tokenizer_name_or_path": ctx_tokenizer.name_or_path,
        "resolved_base_model_name_or_path": state_dict["base_model_name_or_path"],
    }
    write_json(output_dir / "config.json", run_metadata)

    for ds_name in args.datasets:
        logger.info("Loading dataset: %s", ds_name)
        dataset, answers = load_dataset_for_eval(args, model, tokenizer, ctx_tokenizer, ds_name)
        cached_samples = precompute_sample_adapters(model, dataset, tokenizer, device)
        answers_list = answers_for_dataset(ds_name, cached_samples, answers)

        for queue_length in args.queue_lengths:
            logger.info(
                "Evaluating dataset=%s queue_length=%s mode=%s",
                ds_name,
                queue_length,
                run_metadata["mode"],
            )
            metrics, sample_rows = evaluate_queue_length(
                model=model,
                tokenizer=tokenizer,
                cached_samples=cached_samples,
                answers_list=answers_list,
                queue_length=queue_length,
                history_scale=args.history_scale,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            summary_rows.append(
                {
                    "dataset": ds_name,
                    "mode": run_metadata["mode"],
                    "queue_length": queue_length,
                    "history_scale": args.history_scale,
                    **metrics,
                }
            )
            for row in sample_rows:
                row["dataset"] = ds_name
                row["mode"] = run_metadata["mode"]
                row["normalized_label"] = normalize_answer(row["label"])
            per_sample_rows.extend(sample_rows)

            logger.info(
                "Finished dataset=%s queue_length=%s qa_f1=%.4f precision=%.4f recall=%.4f",
                ds_name,
                queue_length,
                metrics["qa_f1_score"],
                metrics["qa_precision"],
                metrics["qa_recall"],
            )

    write_summary_csv(output_dir / "queue_results.csv", summary_rows)
    write_json(output_dir / "queue_results.json", summary_rows)
    write_jsonl(output_dir / "queue_samples.jsonl", per_sample_rows)
    logger.info("Saved summary CSV to %s", output_dir / "queue_results.csv")
    logger.info("Saved per-sample JSONL to %s", output_dir / "queue_samples.jsonl")


if __name__ == "__main__":
    main()
