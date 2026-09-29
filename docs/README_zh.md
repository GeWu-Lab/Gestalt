# Gestalt Stage-III SFT（最小核心版）

本目录只把以下五条路径视为核心接口：

- I2T / MMU 训练；
- T2I 训练；
- I2I 训练；
- MMU（图像 token + 问题 → 文本）推理；
- T2I（文本 → VQVAE token）推理。

模型结构和 checkpoint 参数名保持兼容；渐进数据版本、自定义断点续训、逐样本调试日志及 tokenizer 升级工具不属于核心接口。

## 数据格式

训练输入是 Parquet 文件或包含 Parquet 文件的目录，并且只接受以下统一 schema：

| 字段 | I2T | T2I | I2I |
|---|---|---|---|
| `task_type` | `i2t` | `t2i` | `i2i` |
| `metadata_json` | JSON 字符串 | JSON 字符串 | JSON 字符串 |
| `conversations` | JSON 字符串或 turn 列表 | — | — |
| `img_tokens` | 原始 VQVAE ID | — | — |
| `system_prompt` | 可选 | 字符串 | 字符串 |
| `user_prompt` | — | 字符串 | 字符串 |
| `input_image_tokens` | — | — | 原始 VQVAE ID |
| `answer_image_tokens` | — | 原始 VQVAE ID | 原始 VQVAE ID |

`metadata_json` 必须提供图像 token 网格：

- I2T/T2I：`token_height`、`token_width`；
- I2I：`input_token_height`、`input_token_width`、`output_token_height`、`output_token_width`；
- 多图 I2T 也可使用 `images: [{"token_height": H, "token_width": W}, ...]`。

图像 token 必须全部是 `[0, 16384)` 的原始 VQVAE ID，或全部是 `[126356, 142740)` 的模型空间 ID；两种表示不能混用。I2T 多轮数据固定按 `split_qa` 拆成独立图文问答，防止双向注意力看到未来轮次。

## Tokenizer

训练与推理都直接从模型 checkpoint 目录读取 tokenizer。因此最终 checkpoint（例如
`final_cpt`）必须同时保存模型权重、配置和 `tokenizer.json`。加载时会验证：

- tokenizer 长度为 `142740`；
- 12 个结构/特殊 token 的 ID 与 `gestalt/model/config.py` 完全一致；
- 模型词表扩展到 `142848`，其中 `[142750, 142782)` 是 32 个 interaction token。

当前已核验的参考 tokenizer SHA-256 是
`9ed814b03fb887c8089287493702fc753df11167218ba4b92991bd95970a4c32`。

## 训练

先安装 `requirements.txt`，再在当前 Python 环境中执行：

```bash
DATA_DIR=/path/to/parquet \
PRETRAINED=/path/to/checkpoint \
NPROC_PER_NODE=8 \
scripts/training/gestalt_sft.sh
```

单卡时省略 `NPROC_PER_NODE`。默认使用 `configs/training/gestalt_stage_3.yaml` 和 ZeRO-2；设置 `DEEPSPEED=none` 可关闭 DeepSpeed。数据按 Parquet 文件分配给 rank，因此 Parquet 文件数不能少于分布式 rank 数；不满足时程序会立即报错。

## 推理

MMU：

```bash
python scripts/inference/run_mmu.py \
  --model /path/to/checkpoint \
  --image-tokens image_tokens.npy \
  --question "What is in this image?" \
  --token-h 32 --token-w 32
```

T2I：

```bash
python scripts/inference/run_t2i.py \
  --model /path/to/checkpoint \
  --prompt "A cat sitting on a windowsill" \
  --output output_tokens.npy \
  --lat-h 32 --lat-w 32
```

T2I 的系统提示词默认是空字符串，与训练默认模板一致。若训练数据使用了固定的非空系统提示词，推理时必须通过 `--system-prompt` 传入同一文本。

## 验证

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
GESTALT_CHECKPOINT=/path/to/checkpoint PYTHONPATH=. \
  python -m unittest discover -s tests -p 'test_tokenizer_json.py' -v
python -m compileall -q gestalt scripts tests
bash -n scripts/training/gestalt_sft.sh
```

数据处理测试不需要 GPU；MRoPE、模型 forward/backward 和推理测试需要安装 PyTorch。完整验收还需要一个内含 tokenizer 的真实 checkpoint、小型三任务 Parquet 数据以及 CUDA 环境。
