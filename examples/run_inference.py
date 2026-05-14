import argparse
from pathlib import Path

import torch

from ctx_to_lora.device import get_autocast_context, get_default_device
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel


def parse_args():
    parser = argparse.ArgumentParser(description="Run Doc-to-LoRA inference")
    parser.add_argument("--checkpoint_path", required=True, type=str)
    parser.add_argument(
        "--base_model_path",
        default=None,
        type=str,
        help="Optional local base model path to override checkpoint metadata",
    )
    parser.add_argument("--document_path", required=True, type=str)
    parser.add_argument("--question", required=True, type=str)
    parser.add_argument("--max_new_tokens", default=256, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_default_device()

    print(f"Loading checkpoint from: {args.checkpoint_path}")
    print(f"Running on device: {device}")

    state_dict = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    if args.base_model_path:
        state_dict["base_model_name_or_path"] = args.base_model_path
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict,
        train=False,
        use_flash_attn=device.type == "cuda",
        use_sequence_packing=False,
    )
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    model.reset()

    tokenizer = get_tokenizer(model.base_model.name_or_path)
    doc = Path(args.document_path).read_text(encoding="utf-8")
    chat = [{"role": "user", "content": args.question}]
    chat_ids = tokenizer.apply_chat_template(
        chat,
        add_special_tokens=False,
        return_attention_mask=False,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    model.internalize(doc)
    with torch.inference_mode(), get_autocast_context(device):
        outputs = model.generate(input_ids=chat_ids, max_new_tokens=args.max_new_tokens)

    answer = tokenizer.decode(outputs[0][chat_ids.shape[1] :], skip_special_tokens=True)
    print("\n===== Answer =====\n")
    print(answer.strip())


if __name__ == "__main__":
    main()