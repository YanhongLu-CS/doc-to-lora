# Doc-to-LoRA on Ascend 910B: Gemma Reproduction Summary

This document records the final local code changes and server-side setup used to run `doc-to-lora` inference and evaluation on Huawei Ascend 910B with a local Gemma base model.

## 1. Goal

Run `doc-to-lora` on Ascend 910B for:

- inference
- evaluation
- batched vs iterative comparison
- base-model comparison

The final target line is Gemma-based:

- D2L checkpoint: `trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin`
- local base model: `/home/apulis-dev/userdata/lyh/models/gemma-2-2b-it`

We did **not** target training support or a bit-for-bit recreation of the authors' CUDA environment. We targeted a functionally equivalent inference/evaluation path on Ascend NPU.

## 2. Final Server Environment

We used the existing Conda environment:

- env name: `lyh`

Key runtime facts:

- `torch==2.1.0`
- `torch_npu` available
- `torch.npu.is_available() == True`
- execution happens on Ascend NPU, not CUDA

Important operational choices:

- we did **not** use the repo's original `install.sh`
- we did **not** install CUDA-only packages such as `flash-attn`, `flashinfer`, `bitsandbytes`, `vllm`
- Gemma is loaded from a local model directory, not from Hugging Face at runtime

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

One problematic package was removed:

- `scikit-learn`

Reason:

- it caused `libgomp` / static TLS import failures when `transformers` imported sklearn transitively

## 3. Why Gemma Needed Extra Adaptation

The original Gemma checkpoints store:

- D2L hypernetwork weights
- the upstream base model identifier `google/gemma-2-2b-it`

In the actual server environment:

- online loading through `hf-mirror.com` failed for gated Gemma access
- even after account access was granted, runtime access through the mirror remained unreliable

The practical solution was:

1. download the entire Gemma base model offline or from a machine with working access
2. place it locally at:
   - `/home/apulis-dev/userdata/lyh/models/gemma-2-2b-it`
3. add code support to override the checkpoint's remote base model name with that local path

This is the key Gemma-specific difference from the earlier Qwen path.

## 4. Final Code Changes

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

## 5. What Each Change Did

### 5.1 Device selection and autocast

Added `src/ctx_to_lora/device.py` to centralize:

- default device selection: `npu -> cuda -> cpu`
- device normalization
- default dtype selection
- flash-attention gating
- autocast context selection

This removed hardcoded CUDA assumptions across inference and evaluation.

### 5.2 Model loading compatibility

In `src/ctx_to_lora/model_loading.py`:

- changed default `device` / `dtype` handling to be device-aware
- stopped forcing `device_map="cuda"`
- only enable `flash_attention_2` when truly on CUDA
- only enable `bitsandbytes` quantization when on CUDA
- move models onto NPU explicitly when `device.type == "npu"`

Effect:

- NPU no longer falls into CUDA-only loading branches

### 5.3 Chat template fallback for local model paths

Also in `src/ctx_to_lora/model_loading.py`:

- local model paths now map back to known chat templates
- for example:
  - `/home/.../gemma-2-2b-it` -> `chat_templates/google/gemma-2-2b-it.jinja`

Effect:

- Gemma loaded from a local directory still uses the correct chat template
- avoids fallback to the default template and the `System role not supported` failure

### 5.4 Hypernet compatibility

In `src/ctx_to_lora/modeling/hypernet.py`:

- replaced CUDA-only autocast with device-aware autocast
- made ctx-encoder loading obey the actual runtime device
- gated flash-attention use by device
- guarded `torch.serialization.add_safe_globals` for older PyTorch

Effect:

- `torch==2.1.0` can load checkpoints
- NPU no longer tries to use CUDA-only hypernet code paths

### 5.5 Aggregator / Perceiver compatibility

In `src/ctx_to_lora/modeling/aggregator.py`:

- attention implementation now becomes:
  - `flash_attention_2` if flash-attn exists
  - `eager` otherwise

Effect:

- the Perceiver path no longer assumes flash-attn is always present

### 5.6 Idefics2 Perceiver compatibility

In `src/ctx_to_lora/modeling/idefics2.py`:

- enabled `"eager"` in `IDEFICS2_PERCEIVER_ATTENTION_CLASSES`
- removed the hard assertion that flash-attention must be used
- adapted `Idefics2PerceiverAttention` to support both:
  - cross-attention
  - self-attention
- made `Idefics2PerceiverResampler` support:
  - flash-attn path when available
  - eager attention path otherwise

Effect:

- the context compression path now runs on NPU without `flash-attn`

### 5.7 Torch serialization compatibility

In:

- `src/ctx_to_lora/configs.py`
- `src/ctx_to_lora/modeling/hypernet.py`

we wrapped `torch.serialization.add_safe_globals(...)` with:

```python
if hasattr(torch.serialization, "add_safe_globals"):
```

Effect:

- compatibility with `torch==2.1.0`

### 5.8 Inference script support for local Gemma

Added and extended `examples/run_inference.py`:

- minimal inference entrypoint
- accepts checkpoint/document/question
- now also accepts:
  - `--base_model_path`

Effect:

- Gemma checkpoints can be evaluated with a local base model directory instead of downloading `google/gemma-2-2b-it` online

### 5.9 Evaluation support for local Gemma

In `run_eval.py`:

- added `--base_model_path`

In `src/ctx_to_lora/eval_utils.py`:

- checkpoint loads use `map_location="cpu"`
- model loading is device-aware
- CUDA backend flags are only set when CUDA is present
- `wandb` reporting is disabled with:
  - `report_to = []`
- CSV export tolerates broken or missing `pandas`
- local base model override is propagated through:
  - `run_eval(..., base_model_path=...)`
  - `evaluate(...)`
  - checkpoint state dict override

Effect:

- Gemma evaluation can be run with:
  - a D2L checkpoint
  - a local Gemma base model path
  - no online Gemma fetch

### 5.10 Tokenizer-name normalization during evaluation decode

Also in `src/ctx_to_lora/eval_utils.py`:

- local tokenizer paths are normalized back to canonical model names before indexing `CTX_AFFIXES`

Examples:

- local Gemma path -> `google/gemma-2-2b-it`
- local Qwen path -> `Qwen/Qwen3-4B-Instruct-2507`
- local Mistral path -> `mistralai/Mistral-7B-Instruct-v0.2`

Effect:

- Gemma evaluation no longer crashes in `decode_test_result()` when the tokenizer comes from a local path

### 5.11 Utility compatibility

In `src/ctx_to_lora/utils.py`:

- `clear_gpu()` now works safely for:
  - CUDA
  - NPU
  - non-CUDA environments

### 5.12 Visualization support

Added `scripts/visualize_eval_results.py`:

- no `pandas` dependency
- no `matplotlib` dependency
- merges multiple eval result directories
- generates:
  - `merged_results.csv`
  - `merged_results.json`
  - `report.html`

This is used to compare:

- `base`
- `batch`
- `iterative`

for Gemma runs.

### 5.13 LoRA queue robustness experiment support

Added `scripts/eval_lora_queue.py`:

- precomputes one raw adapter per sample
- constructs a queue of current adapter plus previous `k-1` adapters
- scales history adapters with `--history_scale`
- evaluates robustness as queue length grows
- writes:
  - `queue_results.csv`
  - `queue_results.json`
  - `queue_samples.jsonl`
  - `config.json`

Added `scripts/visualize_lora_queue_results.py`:

- merges one or more queue experiment runs
- plots metrics against queue length
- writes:
  - `merged_queue_results.csv`
  - `merged_queue_results.json`
  - `report.html`

Effect:

- we can now test D2L robustness under accumulated adapter interference

### 5.14 Queue-position experiment support

Added `scripts/eval_lora_queue_positions.py`:

- fixes queue length to a chosen value, typically `4`
- varies the insertion position of the current adapter inside the queue
- writes:
  - `queue_position_results.csv`
  - `queue_position_results.json`
  - `queue_position_samples.jsonl`
  - `config.json`

Added `scripts/visualize_lora_queue_positions.py`:

- visualizes metrics against `recent_position`
- writes:
  - `merged_queue_position_results.csv`
  - `merged_queue_position_results.json`
  - `report.html`

Important interpretation:

- in the current implementation, changing `recent_position` only reorders rank blocks before `combine_lora()`
- because LoRA composition here is ultimately additive across the expanded rank dimension, simple rank-block reordering is mathematically order-invariant
- therefore identical accuracy across positions is expected in this implementation and does **not** by itself indicate a bug

Effect:

- we can explicitly verify that rank-order permutation alone does not change the resulting adapter behavior

## 6. Runtime Issues We Worked Around

### 6.1 sklearn / libgomp TLS failure

Observed:

- `scikit-learn` import caused static TLS / `libgomp` failures

Resolution:

- removed `scikit-learn`

### 6.2 pandas / GLIBCXX mismatch

Observed:

- `pandas` binary expected newer `libstdc++`

Resolution:

- made evaluation CSV export optional when `pandas` is unavailable

### 6.3 Missing evaluation dependencies

Installed when needed:

- `rouge-score`
- `llmlingua`

### 6.4 Dataset warnings

`datasets` emitted warnings around:

- `trust_remote_code`

In our current runs this behaved as a warning/log message and did not block dataset loading after raw data existed locally.

## 7. Data and Models Prepared

### 7.1 Base model

- local Gemma base model:
  - `/home/apulis-dev/userdata/lyh/models/gemma-2-2b-it`

### 7.2 D2L checkpoint

- `trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin`

### 7.3 Evaluation data

- local SQuAD raw data:
  - `data/raw_datasets/squad`

Other datasets such as `drop` and `ropes` were also prepared on the server for the main Gemma runs.

## 8. What Has Been Successfully Reproduced

### 8.1 Gemma inference

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

### 8.2 Gemma evaluation

Successful command pattern:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python run_eval.py \
  --checkpoint_path trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  --base_model_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --datasets squad \
  --split test \
  --eval_batch_size_gen 1 \
  --max_test_samples_per_ds 10 \
  --max_ctx_chunk_len 8192
```

This successfully:

- loaded the checkpoint
- loaded the local Gemma base model
- processed the dataset
- generated evaluation outputs
- decoded outputs
- computed QA metrics

### 8.3 Gemma comparison experiments

The following three Gemma experiment modes have been run successfully:

- `base`
- `batch`
- `iterative`

These runs were then prepared for visualization with `scripts/visualize_eval_results.py`.

### 8.4 Gemma LoRA queue robustness experiment

The Gemma queue robustness experiment has also been run successfully:

- dataset: `squad`
- samples: `50`
- mode: `batch`
- queue lengths:
  - `1`
  - `2`
  - `4`
  - `8`

This experiment evaluates whether the current adapter remains effective when older adapters are also attached.

### 8.5 Gemma queue-position experiment

The Gemma queue-position experiment has also been run:

- dataset: `squad`
- samples: `50`
- mode: `batch`
- queue length: `4`
- recent positions:
  - `0`
  - `1`
  - `2`
  - `3`
- history scale: `0.25`

Observed result:

- `qa_f1_score`, `qa_precision`, and `qa_recall` were effectively identical across all four `recent_position` settings

Interpretation:

- this is expected under the current implementation because the experiment only permutes the order of raw LoRA blocks before concatenation
- after `combine_lora()` and LoRA application, this permutation does not change the final additive update
- so this experiment verifies order-invariance of the current queue composition, rather than exposing a new robustness difference

## 9. What Is Reproduced vs. What Is Not

### 9.1 Reproduced now

- Gemma-based D2L inference
- Gemma-based D2L evaluation
- Gemma `base` vs `batch` vs `iterative` comparison
- HTML result visualization from saved eval runs

### 9.2 Not in current scope

- training
- exact recreation of the authors' original CUDA performance characteristics
- unsupported CUDA-only baselines that still assume the original stack end-to-end

## 10. Is Functionality Reduced?

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

## 11. Recommended Reproducible Gemma Experiments

Best supported now:

1. Single-document Gemma D2L inference
2. Gemma D2L QA evaluation on:
   - `squad`
   - `drop`
   - `ropes`
3. Gemma `base` vs `batch` vs `iterative` comparison
4. Gemma LoRA queue robustness evaluation
5. Gemma queue-position order-invariance check
6. HTML visualization of the comparison and queue results

## 12. Experiment Commands

This section lists the concrete script commands used for each experiment class.

### 12.1 Gemma single-document inference

Script:

- `examples/run_inference.py`

Command:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python examples/run_inference.py \
  --checkpoint_path trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  --base_model_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --document_path data/sakana_wiki.txt \
  --question "Tell me about Sakana AI."
```

### 12.2 Gemma D2L small evaluation sanity check

Script:

- `run_eval.py`

Command:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python run_eval.py \
  --checkpoint_path trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  --base_model_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --datasets squad \
  --split test \
  --eval_batch_size_gen 1 \
  --max_test_samples_per_ds 10 \
  --max_ctx_chunk_len 8192
```

### 12.3 Gemma D2L batch evaluation

Script:

- `run_eval.py`

Command:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python run_eval.py \
  --checkpoint_path trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  --base_model_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --datasets squad drop ropes \
  --split test \
  --eval_batch_size_gen 1 \
  --max_test_samples_per_ds 50 \
  --max_ctx_chunk_len 8192
```

### 12.4 Gemma D2L iterative evaluation

Script:

- `run_eval.py`

Command:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python run_eval.py \
  --checkpoint_path trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  --base_model_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --datasets squad drop ropes \
  --split test \
  --eval_batch_size_gen 1 \
  --max_test_samples_per_ds 50 \
  --max_ctx_chunk_len 8192 \
  --use_iterative_mode
```

### 12.5 Gemma base-model evaluation

Script:

- `run_eval.py`

Command:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python run_eval.py \
  --model_name_or_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --datasets squad drop ropes \
  --split test \
  --eval_batch_size_gen 1 \
  --max_test_samples_per_ds 50 \
  --max_ctx_chunk_len 8192
```

### 12.6 Gemma base vs batch vs iterative visualization

Script:

- `scripts/visualize_eval_results.py`

Command template:

```bash
python scripts/visualize_eval_results.py \
  --run batch=/path/to/batch_run_dir \
  --run iterative=/path/to/iterative_run_dir \
  --run base=/path/to/base_run_dir \
  --output-dir /path/to/output_dir
```

### 12.7 Gemma LoRA queue robustness evaluation

Script:

- `scripts/eval_lora_queue.py`

Command:

```bash
python scripts/eval_lora_queue.py \
  --checkpoint_path trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  --base_model_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --datasets squad \
  --split test \
  --max_test_samples_per_ds 50 \
  --max_ctx_chunk_len 8192 \
  --queue_lengths 1 2 4 8 \
  --history_scale 0.25
```

### 12.8 Gemma LoRA queue iterative robustness evaluation

Script:

- `scripts/eval_lora_queue.py`

Command:

```bash
python scripts/eval_lora_queue.py \
  --checkpoint_path trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  --base_model_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --datasets squad \
  --split test \
  --max_test_samples_per_ds 50 \
  --max_ctx_chunk_len 8192 \
  --queue_lengths 1 2 4 8 \
  --history_scale 0.25 \
  --use_iterative_mode
```

### 12.9 Gemma LoRA queue visualization

Script:

- `scripts/visualize_lora_queue_results.py`

Command template:

```bash
python scripts/visualize_lora_queue_results.py \
  --run batch=/path/to/batch_queue_run_dir \
  --run iterative=/path/to/iterative_queue_run_dir \
  --output-dir /path/to/output_dir
```

### 12.10 Gemma queue-position experiment

Script:

- `scripts/eval_lora_queue_positions.py`

Command:

```bash
python scripts/eval_lora_queue_positions.py \
  --checkpoint_path trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  --base_model_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --datasets squad \
  --split test \
  --max_test_samples_per_ds 50 \
  --max_ctx_chunk_len 8192 \
  --queue_length 4 \
  --recent_positions 0 1 2 3 \
  --history_scale 0.25
```

### 12.11 Gemma queue-position visualization

Script:

- `scripts/visualize_lora_queue_positions.py`

Command template:

```bash
python scripts/visualize_lora_queue_positions.py \
  --run batch=/path/to/batch_queue_position_run_dir \
  --run iterative=/path/to/iterative_queue_position_run_dir \
  --output-dir /path/to/output_dir
```

## 13. Visualization Workflow

The comparison script can be used like this:

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

The generated HTML report is intended as the main quick-look summary for experiment comparison.

For queue experiments, the queue visualization script writes:

- `merged_queue_results.csv`
- `merged_queue_results.json`
- `report.html`

The queue report is intended to visualize how `qa_f1_score`, `qa_precision`, `qa_recall`, runtime, and throughput change as queue length increases.

For queue-position experiments, the position visualization script writes:

- `merged_queue_position_results.csv`
- `merged_queue_position_results.json`
- `report.html`

The queue-position report is intended to visualize how metrics change with `recent_position`. In the current implementation, identical accuracy across positions should be interpreted as expected order-invariance, not necessarily as an implementation defect.

## 14. Current Bottom Line

At this point, the repository has been adapted so that:

- Ascend 910B inference works
- Ascend 910B evaluation works
- Gemma can be loaded from a local base model directory
- Gemma `base`, `batch`, and `iterative` experiments are reproducible
- Gemma LoRA queue robustness experiments are reproducible
- Gemma queue-position experiments are reproducible
- result visualization is available without hardcoded paths

The main remaining gap is not basic Gemma usability. The main remaining gap is only broader experiment coverage beyond the currently supported inference/evaluation workflow.
