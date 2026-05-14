# Doc-to-LoRA on Ascend 910B: Reproduction Summary

This document records what we changed from the upstream repository, what was configured on the server, what has already been reproduced successfully, and what remains out of scope.

## 1. Goal

Run `doc-to-lora` on a Huawei Ascend 910B-64G server for:

- inference only
- evaluation only
- no training

We support both Gemma and Qwen checkpoints:
- Gemma: `trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin`
- Qwen: `trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin`

We targeted a functionally equivalent inference and evaluation path on Ascend NPU.

## 2. Server-side Environment

We used the existing Python environment:

- `torch==2.1.0`
- `torch_npu` available
- `torch.npu.is_available() == True`

Important operational choices:

- we did **not** use the repo's original `install.sh`
- we did **not** install CUDA packages such as `flash-attn`, `flashinfer`, `bitsandbytes`, `vllm`
- Gemma is loaded from a local model directory when needed

Packages explicitly installed during setup included:

- `transformers==4.51.3`
- `accelerate==1.6.0`
- `peft`
- `einops`
- `jaxtyping`
- `sentencepiece`
- `datasets`
- `huggingface-hub`
- `rouge-score`
- `llmlingua`

We also removed one problematic package:

- `scikit-learn`

Reason: it caused `libgomp` / static TLS import failures when `transformers` imported sklearn transitively.

## 3. Why Gemma Needed Extra Adaptation

The original Gemma checkpoints store:

- D2L hypernetwork weights
- the upstream base model identifier `google/gemma-2-2b-it`

In some server environments:

- online loading through HuggingFace may fail for gated Gemma access
- the practical solution is to download the entire Gemma base model locally first
- place it locally at: `/home/apulis-dev/userdata/lyh/models/gemma-2-2b-it`
- add code support to override the checkpoint's remote base model name with that local path

This is the key Gemma-specific difference from the earlier Qwen path.

## 4. Code Changes Made

### 4.1 New files

- `src/ctx_to_lora/device.py`
- `examples/run_inference.py`
- `scripts/eval_lora_queue.py`
- `scripts/visualize_eval_results.py`
- `scripts/visualize_lora_queue_results.py`
- `scripts/eval_lora_queue_positions.py`
- `scripts/visualize_lora_queue_positions.py`
- `docs/ASCEND_910B_INFERENCE.md`
- `docs/ASCEND_910B_REPRO_SUMMARY.md`
- `docs/INFERENCE.md`

### 4.2 Updated files

- `README.md`
- `demo/app.py`
- `run_eval.py`
- `src/ctx_to_lora/configs.py`
- `src/ctx_to_lora/eval_utils.py`
- `src/ctx_to_lora/model_loading.py`
- `src/ctx_to_lora/modeling/aggregator.py`
- `src/ctx_to_lora/modeling/hypernet.py`
- `src/ctx_to_lora/modeling/idefics2.py`
- `src/ctx_to_lora/utils.py`
- `train.py`

### 4.3 Device selection

Added `src/ctx_to_lora/device.py` to centralize:

- default device selection: `npu -> cuda -> cpu`
- device normalization
- default dtype selection
- flash-attention gating
- autocast context selection

This removed hardcoded CUDA assumptions across inference and evaluation.

### 4.4 Model loading compatibility

In `src/ctx_to_lora/model_loading.py`:

- changed default `device`/`dtype` handling to be device-aware
- stopped forcing `device_map="cuda"`
- only enable `flash_attention_2` when truly on CUDA
- only enable `bitsandbytes` quantization when on CUDA
- move models onto NPU explicitly when `device.type == "npu"`

Effect: NPU no longer falls into CUDA-only loading branches.

### 4.5 Chat template fallback for local model paths

Also in `src/ctx_to_lora/model_loading.py`:

- local model paths now map back to known chat templates
- for example: `/home/.../gemma-2-2b-it` -> `chat_templates/google/gemma-2-2b-it.jinja`

Effect: Gemma loaded from a local directory still uses the correct chat template.

### 4.6 Hypernet compatibility

In `src/ctx_to_lora/modeling/hypernet.py`:

- replaced CUDA-only autocast with device-aware autocast
- made ctx-encoder loading obey the actual runtime device
- gated flash-attention use by device
- guarded `torch.serialization.add_safe_globals` for older PyTorch

Effect: `torch==2.1.0` can load checkpoints, NPU no longer tries to use CUDA-only hypernet code paths.

### 4.7 Aggregator / Perceiver compatibility

In `src/ctx_to_lora/modeling/aggregator.py`:

- attention implementation now becomes: `flash_attention_2` if flash-attn exists, `eager` otherwise

Effect: the Perceiver path no longer assumes flash-attn is always present.

### 4.8 Idefics2 Perceiver compatibility

In `src/ctx_to_lora/modeling/idefics2.py`:

- enabled `"eager"` in `IDEFICS2_PERCEIVER_ATTENTION_CLASSES`
- removed the hard assertion that flash-attention must be used
- adapted `Idefics2PerceiverAttention` to support both cross-attention and self-attention
- made `Idefics2PerceiverResampler` support flash-attn path when available and eager attention path otherwise

Effect: the context compression path now runs on NPU without `flash-attn`.

### 4.9 Torch serialization compatibility

In `src/ctx_to_lora/configs.py` and `src/ctx_to_lora/modeling/hypernet.py`:

- wrapped `torch.serialization.add_safe_globals(...)` with `hasattr(torch.serialization, "add_safe_globals")`

Effect: compatibility with `torch==2.1.0`.

### 4.10 Inference script support for local models

Added and extended `examples/run_inference.py`:

- minimal inference entrypoint
- accepts checkpoint/document/question
- accepts `--base_model_path` for local model paths

Effect: Gemma/Qwen checkpoints can be evaluated with a local base model directory.

### 4.11 Evaluation support for local models

In `run_eval.py`:

- added `--base_model_path`

In `src/ctx_to_lora/eval_utils.py`:

- checkpoint loads use `map_location="cpu"`
- model loading is device-aware
- CUDA backend flags are only set when CUDA is present
- `wandb` reporting is disabled with `report_to = []`
- CSV export tolerates broken or missing `pandas`
- local base model override is propagated through the evaluation pipeline

Effect: Gemma/Qwen evaluation can be run with a local base model path without online fetches.

### 4.12 Tokenizer-name normalization during evaluation decode

Also in `src/ctx_to_lora/eval_utils.py`:

- local tokenizer paths are normalized back to canonical model names before indexing `CTX_AFFIXES`

Examples:

- local Gemma path -> `google/gemma-2-2b-it`
- local Qwen path -> `Qwen/Qwen3-4B-Instruct-2507`
- local Mistral path -> `mistralai/Mistral-7B-Instruct-v0.2`

Effect: evaluation no longer crashes when the tokenizer comes from a local path.

### 4.13 Demo and simple inference

In `demo/app.py`:

- device selection is now NPU-aware
- flash-attn is only used on CUDA
- autocast uses the runtime device

Added `examples/run_inference.py`:

- minimal CLI inference entrypoint
- accepts checkpoint/document/question
- loads on NPU successfully

### 4.14 Evaluation compatibility

In `src/ctx_to_lora/eval_utils.py`:

- made model loading device-aware
- avoided CUDA-specific backend configuration when CUDA is absent
- added `map_location="cpu"` when loading checkpoints for evaluation
- disabled `wandb` reporting via `eval_trainer_args["report_to"] = []`
- made CSV export optional when `pandas` is unavailable or broken

In `src/ctx_to_lora/utils.py`:

- made `clear_gpu()` safe for CUDA, NPU, and non-CUDA environments

Effect: `run_eval.py` now works on the Ascend inference environment.

## 5. Runtime / Data Issues We Worked Around

### 5.1 sklearn / libgomp TLS failure

Observed: `scikit-learn` import caused static TLS / `libgomp` failures

Resolution: removed `scikit-learn`

### 5.2 pandas / GLIBCXX mismatch

Observed: `pandas` binary expected newer `libstdc++`

Resolution: made evaluation CSV export optional when `pandas` is unavailable

### 5.3 Missing evaluation dependencies

Installed when needed:

- `rouge-score`
- `llmlingua`

### 5.4 Dataset warnings

`datasets` emitted warnings around `trust_remote_code`. In our current runs this behaved as a warning and did not block dataset loading after the raw data existed locally.

## 6. Data and Models Prepared

### 6.1 Base models

- local Gemma base model: `/home/apulis-dev/userdata/lyh/models/gemma-2-2b-it`
- local Qwen base model: `/home/apulis-dev/userdata/lyh/models/Qwen3-4B`

### 6.2 D2L checkpoints

- Gemma: `trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin`
- Qwen: `trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin`
- Mistral: `trained_d2l/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin`

### 6.3 Evaluation data

- local SQuAD raw data: `data/raw_datasets/squad`

Other datasets such as `drop` and `ropes` require network access to download.

## 7. What Has Been Successfully Reproduced

### 7.1 Gemma inference

Successful command pattern:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python examples/run_inference.py \
  --checkpoint_path trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  --base_model_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --document_path data/sakana_wiki.txt \
  --question "Tell me about Sakana AI."
```

This successfully:

- loaded the Gemma checkpoint
- loaded the local Gemma base model
- internalized the document
- generated an answer on NPU

### 7.2 Qwen inference

```bash
python examples/run_inference.py \
  --checkpoint_path trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin \
  --base_model_path /home/apulis-dev/userdata/lyh/models/Qwen3-4B \
  --document_path data/sakana_wiki.txt \
  --question "Tell me about Sakana AI."
```

### 7.3 Evaluation

Successful small-sample evaluation on:

- `squad` (local)
- `drop` (requires network)
- `ropes` (requires network)

Example command:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python run_eval.py \
  --checkpoint_path trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  --base_model_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --datasets squad \
  --split test \
  --eval_batch_size_gen 1 \
  --max_test_samples_per_ds 50 \
  --max_ctx_chunk_len 8192
```

## 8. What Is Reproduced vs. What Is Not

### 8.1 Reproduced now

- Gemma-based D2L inference
- Gemma-based D2L evaluation
- Qwen-based D2L inference
- Qwen-based D2L evaluation
- Gemma `base` vs `batch` vs `iterative` comparison
- HTML result visualization from saved eval runs

### 8.2 Not in current scope

- training
- exact recreation of the authors' original CUDA performance characteristics
- unsupported CUDA-only baselines that still assume the original stack end-to-end

## 9. Is Functionality Reduced?

For the current NPU inference/evaluation target:

- no deliberate removal of D2L inference functionality
- no deliberate removal of D2L evaluation functionality
- no deliberate change to the main checkpoint interfaces

What changed is backend behavior:

- CUDA-only accelerations were disabled or replaced by eager implementations
- local base model paths are supported in addition to upstream Hugging Face names
- some optional export/report paths were made more forgiving

So this should be described as:

- **compatibility adaptation**
- **not a functional rewrite**
- **not a feature amputation for inference/evaluation**

What is different from the original CUDA path:

- performance characteristics
- memory behavior
- some optional baseline/training workflows remain out of scope

## 10. Recommended Reproducible Experiments

Best supported now:

1. Single-document Gemma/Qwen D2L inference
2. Gemma/Qwen D2L QA evaluation on `squad`
3. Gemma/Qwen `base` vs `batch` vs `iterative` comparison
4. HTML visualization of the comparison results

## 11. Experiment Commands

### 11.1 Gemma single-document inference

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python examples/run_inference.py \
  --checkpoint_path trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  --base_model_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --document_path data/sakana_wiki.txt \
  --question "Tell me about Sakana AI."
```

### 11.2 Qwen single-document inference

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python examples/run_inference.py \
  --checkpoint_path trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin \
  --base_model_path /home/apulis-dev/userdata/lyh/models/Qwen3-4B \
  --document_path data/sakana_wiki.txt \
  --question "Tell me about Sakana AI."
```

### 11.3 Local evaluation (squad only)

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python run_eval.py \
  --checkpoint_path trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  --base_model_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --datasets squad \
  --split test \
  --eval_batch_size_gen 1 \
  --max_test_samples_per_ds 50 \
  --max_ctx_chunk_len 8192
```

### 11.4 Base model evaluation (without D2L)

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python run_eval.py \
  --model_name_or_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --datasets squad \
  --split test \
  --eval_batch_size_gen 1 \
  --max_test_samples_per_ds 50
```

### 11.5 Visualization

```bash
python scripts/visualize_eval_results.py \
  --run batch=/path/to/batch_run_dir \
  --run iterative=/path/to/iterative_run_dir \
  --run base=/path/to/base_run_dir \
  --output-dir /path/to/output_dir
```

Outputs:

- `merged_results.csv`
- `merged_results.json`
- `report.html`

## 12. Current Bottom Line

At this point, the repository has been adapted so that:

- Ascend 910B inference works for Gemma and Qwen
- Ascend 910B evaluation works
- Gemma/Qwen can be loaded from a local base model directory
- Gemma/Qwen `base`, `batch`, and `iterative` experiments are reproducible
- result visualization is available without hardcoded paths

The main remaining gap is broader experiment coverage beyond the currently supported inference/evaluation workflow.