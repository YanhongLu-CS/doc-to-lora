# Doc-to-LoRA on Ascend 910B (Inference Only)

This guide is for running inference only on Huawei Ascend 910B 64G cards.

## 1. What changes on 910B

The upstream project is written for NVIDIA CUDA by default:

- `install.sh` installs CUDA wheels, `flash-attn`, and `flashinfer`
- `pyproject.toml` includes packages such as `vllm`, `bitsandbytes`, and `liger-kernel`
- several code paths assumed `cuda`

For Ascend inference, use a minimal inference environment instead of `./install.sh` or `uv sync`.

## 2. Server prerequisites

Make sure the server already has:

- Ascend driver installed
- CANN installed
- a Python 3.10 environment
- an officially matched `torch` + `torch_npu` pair for your CANN release

Basic verification:

```bash
npu-smi info
python -c "import torch; import torch_npu; print(torch.npu.is_available())"
```

If `True` is printed, the runtime is visible to PyTorch.

## 3. Create a clean Python environment

Example with `venv`:

```bash
cd /path/to/doc-to-lora
python3.10 -m venv .venv
source .venv/bin/activate
python -V
```

Install the Ascend PyTorch stack first using the official wheel pair that matches your CANN version.

Then install only the packages needed for inference:

```bash
pip install transformers==4.51.3 accelerate==1.6.0 peft einops jaxtyping \
  gradio>=4.40.0 flask pandas plotly datasets huggingface-hub sentencepiece
pip install -e . --no-deps
```

Do not install these for 910B inference:

- `flash-attn`
- `flashinfer-python`
- `vllm`
- `bitsandbytes`
- `liger-kernel`
- `deepspeed`

## 4. Download the pretrained checkpoint

```bash
huggingface-cli login
huggingface-cli download SakanaAI/doc-to-lora \
  --local-dir trained_d2l \
  --include "*/"
```

After download, confirm the checkpoint exists:

```bash
find trained_d2l -name pytorch_model.bin
```

## 5. Run a first smoke test

Use the lightweight inference script in this repo:

```bash
python examples/run_inference.py \
  --checkpoint_path trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  --document_path data/sakana_wiki.txt \
  --question "Tell me about Sakana AI."
```

If you want to use your own document:

```bash
python examples/run_inference.py \
  --checkpoint_path trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  --document_path /path/to/your_doc.txt \
  --question "请根据文档回答：这份文档的核心内容是什么？"
```

## 6. Optional: start the Gradio demo

```bash
python demo/app.py
```

If the server is remote, bind or forward the port in the usual way before opening the page in your browser.

## 7. Recommended single-card first run

Even if your server has 2 cards, start with one card first:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python examples/run_inference.py \
  --checkpoint_path trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  --document_path data/sakana_wiki.txt \
  --question "Tell me about Sakana AI."
```

After one-card inference works, you can test card 1:

```bash
export ASCEND_RT_VISIBLE_DEVICES=1
```

For this inference-only path, dual-card execution is not required. One 910B-64G card is usually the easier and safer starting point.

## 8. Common issues

### `ModuleNotFoundError: torch_npu`

Your Ascend PyTorch environment is incomplete. Install the official `torch_npu` wheel that matches the installed CANN release.

### `flash_attention_2` or CUDA-related errors

You accidentally installed CUDA-only dependencies or used the original CUDA installation flow. Remove the environment and recreate it with the minimal inference package set above.

### `No checkpoints found`

The checkpoint download did not land under `trained_d2l/`, or the path is different from the example. Locate the real `pytorch_model.bin` and pass it explicitly.

### OOM on generation

Try:

- shorter documents
- smaller `--max_new_tokens`
- single-card inference
- `ASCEND_RT_VISIBLE_DEVICES=0`

## 9. What is already adjusted in this repo

This repo now contains:

- automatic device selection for `npu` / `cuda` / `cpu`
- a minimal inference script at `examples/run_inference.py`
- demo loading that avoids CUDA-only flash attention outside NVIDIA GPUs

That means the remaining work on your server is mostly environment setup rather than code surgery.
