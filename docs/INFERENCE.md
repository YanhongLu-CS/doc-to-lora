# Doc-to-LoRA 推理实验指南

本文档介绍如何使用本地模型进行 Doc-to-LoRA 推理实验。

## 环境准备

```bash
cd /home/apulis-dev/userdata/lyh/Reproduction/doc-to-lora
source .venv/bin/activate
```

## 可用的 Checkpoint

| Checkpoint 路径 | 对应基础模型 |
|----------------|-------------|
| `trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin` | gemma-2-2b-it |
| `trained_d2l/gemma_2b_d2l/checkpoint-20000/pytorch_model.bin` | gemma-2-2b-it |
| `trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin` | Qwen2.5-4B-Instruct |
| `trained_d2l/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin` | Mistral-7B |

本地基础模型路径：`/home/apulis-dev/userdata/lyh/models/`

---

## 1. 基础推理命令

使用 `examples/run_inference.py` 进行简单的上下文+问答推理。

### 命令模板

```bash
python examples/run_inference.py \
  --checkpoint_path /home/apulis-dev/userdata/lyh/Reproduction/doc-to-lora/trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  --base_model_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --document_path /path/to/your_context.txt \
  --question "Your question here" \
  --max_new_tokens 256
```

### 示例

```bash
python examples/run_inference.py \
  --checkpoint_path /home/apulis-dev/userdata/lyh/Reproduction/doc-to-lora/trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  --base_model_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --document_path /home/apulis-dev/userdata/lyh/Reproduction/doc-to-lora/data/sakana_wiki.txt \
  --question "Tell me about Sakana AI."
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--checkpoint_path` | D2L checkpoint 路径（必填） |
| `--base_model_path` | 本地基础模型路径（必填，避免联网） |
| `--document_path` | 上下文文档路径（必填） |
| `--question` | 你的问题（必填） |
| `--max_new_tokens` | 最大生成 token 数，默认 256 |

---

## 2. 使用其他 Checkpoint

### Qwen-4B 模型

```bash
python examples/run_inference.py \
  --checkpoint_path /home/apulis-dev/userdata/lyh/Reproduction/doc-to-lora/trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin \
  --base_model_path /home/apulis-dev/userdata/lyh/models/Qwen3-4B \
  --document_path /path/to/your_context.txt \
  --question "Your question here"
```

### Mistral-7B 模型

```bash
python examples/run_inference.py \
  --checkpoint_path /home/apulis-dev/userdata/lyh/Reproduction/doc-to-lora/trained_d2l/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin \
  --base_model_path /home/apulis-dev/userdata/lyh/models/mistral_7b \
  --document_path /path/to/your_context.txt \
  --question "Your question here"
```

---

## 3. 交互式 Demo

启动 Gradio 网页界面，可以手动输入上下文和问题：

```bash
python demo/app.py
```

访问 `http://localhost:7861`

界面功能：
- 左侧：输入上下文（Context）和调整上下文强度（Context Scaling）
- 右侧：输入问题进行对话

---

## 4. 标准评估实验

使用 `run_eval.py` 在标准数据集上进行评估。

> **注意**：使用 `run_eval.py` 评估需要下载数据集到本地。当前只有 `squad` 数据集在本地可用（位于 `data/raw_datasets/squad/`）。其他数据集（`drop`、`ropes`、`longbench` 等）需要联网下载。

### 本地数据集评估（不联网）

如果网络不可用，只能使用本地已有的 `squad` 数据集：

```bash
# 基础模型评估
python run_eval.py \
  --model_name_or_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --datasets squad \
  --split test \
  --eval_batch_size_gen 1

# D2L 模型推理
python run_eval.py \
  --checkpoint_path /home/apulis-dev/userdata/lyh/Reproduction/doc-to-lora/trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  --base_model_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --datasets squad \
  --split test \
  --max_ctx_chunk_len 8192 \
  --eval_batch_size_gen 1
```

### 联网评估（需要网络连接）

如果可以联网，可以使用更多数据集：

#### 基础模型评估（无 D2L adapter）

```bash
python run_eval.py \
  --model_name_or_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --datasets squad drop ropes \
  --split test \
  --eval_batch_size_gen 1
```

#### D2L 模型推理

```bash
python run_eval.py \
  --checkpoint_path /home/apulis-dev/userdata/lyh/Reproduction/doc-to-lora/trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  --base_model_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --datasets squad drop ropes longbench/qasper_e longbench/2wikimqa_e longbench/multifieldqa_en_e \
  --split test \
  --max_ctx_chunk_len 8192 \
  --eval_batch_size_gen 1
```

### 迭代模式（逐层生成 LoRA）

```bash
python run_eval.py \
  --checkpoint_path /home/apulis-dev/userdata/lyh/Reproduction/doc-to-lora/trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  --base_model_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --datasets squad drop ropes \
  --split test \
  --max_ctx_chunk_len 8192 \
  --eval_batch_size_gen 1 \
  --use_iterative_mode
```

### 无上下文（消融实验）

```bash
python run_eval.py \
  --model_name_or_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --datasets squad drop ropes longbench/qasper_e longbench/2wikimqa_e longbench/multifieldqa_en_e \
  --split test \
  --eval_batch_size_gen 1 \
  --remove_context
```

### Context Distillation (CD) 推理

```bash
python run_eval.py \
  --model_name_or_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --datasets squad drop ropes \
  --split test \
  --use_cd \
  --cd_update_iterations 300 \
  --eval_batch_size_gen=1 \
  --truncate_if_too_long_inp
```

### LLMLingua 压缩推理

```bash
python run_eval.py \
  --model_name_or_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --datasets squad drop ropes \
  --split test \
  --eval_batch_size_gen=1 \
  --use_llmlingua \
  --llmlingua_compression_rate 0.8 \
  --truncate_if_too_long_ctx
```

### Text-to-LoRA 推理

```bash
# 先下载 T2L checkpoint
python -m huggingface_hub.commands.huggingface_cli download \
  SakanaAI/text-to-lora \
  --local-dir . \
  --include "trained_t2l/gemma_2b_t2l"

# 使用 T2L 推理
python run_eval.py \
  --model_name_or_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it \
  --datasets squad drop ropes \
  --split test \
  --eval_batch_size_gen=1 \
  --use_t2l
```

---

## 5. 评估数据集说明

| 数据集 | 说明 |
|--------|------|
| `squad` | Stanford Question Answering Dataset |
| `drop` | Discrete Reasoning Over Paragraphs |
| `ropes` | Reasoning Over Paragraphs |
| `longbench/qasper_e` | LongBench QASPER |
| `longbench/2wikimqa_e` | LongBench 2WikiMQA |
| `longbench/multifieldqa_en_e` | LongBench MultiFieldQA |

---

## 6. 常见问题

### 网络不可达错误

如果出现 `Network is unreachable` 错误，确保添加 `--base_model_path` 参数指向本地模型：

```bash
--base_model_path /home/apulis-dev/userdata/lyh/models/gemma-2-2b-it
```

### OOM（内存不足）

尝试：
- 减小 `--max_new_tokens`
- 使用更短的上下文文档
- 单卡推理时设置 `export ASCEND_RT_VISIBLE_DEVICES=0`

### 自定义 Checkpoint

如果使用自己训练的 checkpoint，确保路径正确：

```bash
--checkpoint_path /path/to/your/checkpoint/pytorch_model.bin
--base_model_path /home/apulis-dev/userdata/lyh/models/your-base-model
```
