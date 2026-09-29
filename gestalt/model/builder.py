"""Model utilities used by Stage-III SFT training."""

import importlib
import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import PreTrainedModel


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 词表扩展
# ---------------------------------------------------------------------------

def extend_llada_vocab(
    model: PreTrainedModel,
    target_vocab_size: int,
    new_embedding_init: str = "mean",
) -> None:
    """
    将 LLaDA 模型的词表从当前大小扩展到 target_vocab_size。

    该函数同时扩展输入 embedding 层（wte）和输出投影层（ff_out），
    并使用 mean-of-existing-embeddings 策略初始化新增的 embedding 向量，
    使用 zero-normal 策略初始化新增的输出投影权重。

    Args:
        model: 已加载的 LLaDA 模型实例（PreTrainedModel）
        target_vocab_size: 目标模型词表大小（16K VQ 配置为 142848）
        new_embedding_init: 新 embedding 初始化策略
            - "mean": 用现有 embedding 均值初始化（默认，训练稳定）
            - "zero": 全零初始化
    """
    wte = model.get_input_embeddings()
    current_size = wte.weight.shape[0]

    if current_size >= target_vocab_size:
        logger.info(
            f"词表已满足要求，无需扩展。当前大小：{current_size}，目标大小：{target_vocab_size}"
        )
        return

    logger.info(f"扩展词表：{current_size} → {target_vocab_size}")

    # ── 1. 扩展输入 Embedding（wte）────────────────────────────────────────
    d_model = wte.weight.shape[1]
    device = wte.weight.device
    dtype = wte.weight.dtype

    new_wte = nn.Embedding(target_vocab_size, d_model, device=device, dtype=dtype)

    with torch.no_grad():
        # 复制原有 embedding
        new_wte.weight.data[:current_size] = wte.weight.data[:current_size]

        # 初始化新增 embedding
        num_new = target_vocab_size - current_size
        if new_embedding_init == "mean":
            mean_emb = wte.weight.data.mean(dim=0, keepdim=True)
            new_wte.weight.data[current_size:] = mean_emb.expand(num_new, -1)
        else:  # "zero"
            new_wte.weight.data[current_size:].zero_()

    model.set_input_embeddings(new_wte)

    # ── 2. 扩展输出投影层（ff_out）──────────────────────────────────────────
    # LLaDA 结构：model.model.transformer.ff_out
    if not (hasattr(model, "model")
            and hasattr(model.model, "transformer")
            and hasattr(model.model.transformer, "ff_out")):
        raise AttributeError(
            "无法找到 LLaDA 的输出投影层 model.model.transformer.ff_out，"
            "请确认模型结构是否为 LLaDA-8B。"
        )

    old_ff_out = model.model.transformer.ff_out
    old_vocab_size = old_ff_out.weight.shape[0]
    d_model = old_ff_out.weight.shape[1]
    device = old_ff_out.weight.device
    dtype = old_ff_out.weight.dtype

    new_ff_out = nn.Linear(d_model, target_vocab_size, bias=False, device=device, dtype=dtype)

    with torch.no_grad():
        new_ff_out.weight.data[:old_vocab_size] = old_ff_out.weight.data[:old_vocab_size]
        if old_vocab_size < target_vocab_size:
            nn.init.normal_(
                new_ff_out.weight.data[old_vocab_size:],
                mean=0.0,
                std=0.02,
            )

    model.model.transformer.ff_out = new_ff_out

    # ── 3. 更新 config ──────────────────────────────────────────────────────
    model.config.vocab_size = target_vocab_size
    # LLaDA config 里 embedding_size 是独立字段，训练时必须与 vocab_size 同步；
    # 否则下次从 checkpoint 初始化时会触发 "embedding size should be at least as big as vocab size"
    if hasattr(model.config, "embedding_size"):
        model.config.embedding_size = target_vocab_size

    logger.info(
        f"词表扩展完成。\n"
        f"  wte    : {model.get_input_embeddings().weight.shape}\n"
        f"  ff_out : {model.model.transformer.ff_out.weight.shape}"
    )


# ---------------------------------------------------------------------------
# Gradient Checkpointing
# ---------------------------------------------------------------------------

def enable_llada_gradient_checkpointing(model: PreTrainedModel) -> None:
    """
    为 LLaDA 模型启用 Gradient Checkpointing。

    LLaDA 使用自定义的 ActivationCheckpointingStrategy，无法直接调用标准
    HuggingFace 的 model.gradient_checkpointing_enable()，需要通过
    动态注入的方式注册兼容接口。

    Args:
        model: 已加载的 LLaDA 模型实例
    """
    cfg_modname = model.config.__class__.__module__
    mdl_modname = model.__class__.__module__

    cfg_mod = importlib.import_module(cfg_modname)
    mdl_mod = importlib.import_module(mdl_modname)

    ActivationCheckpointingStrategy = cfg_mod.ActivationCheckpointingStrategy

    logger.info(
        f"注册 GC 兼容接口。\n"
        f"  config module : {cfg_modname}\n"
        f"  model module  : {mdl_modname}\n"
        f"  strategy enum : {ActivationCheckpointingStrategy}"
    )

    def _gc_enable(self, gradient_checkpointing_kwargs=None):
        self.config.use_cache = False
        self.model.set_activation_checkpointing(
            ActivationCheckpointingStrategy.whole_layer
        )

    setattr(mdl_mod.LLaDAModelLM, "supports_gradient_checkpointing", True)
    setattr(mdl_mod.LLaDAModelLM, "gradient_checkpointing_enable", _gc_enable)

    logger.info("Gradient Checkpointing 接口注册完成（whole_layer 策略）。")


# ---------------------------------------------------------------------------
# 新增参数重初始化
# ---------------------------------------------------------------------------

def _reinit_missing_parameters(
    model: PreTrainedModel,
    missing_keys,
) -> None:
    """Initialize only explicitly missing, known Gestalt parameters.

    Vision MLP parameters are copied from the corresponding text MLP. MRoPE
    frequency buffers are deterministic and are recomputed. Any other missing
    key is treated as a checkpoint incompatibility instead of being silently
    randomized.
    """
    missing = set(missing_keys)
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    handled = set()
    vision_name_map = {
        ".vision_ff_proj.": ".ff_proj.",
        ".vision_up_proj.": ".up_proj.",
        ".vision_ff_out.": ".ff_out.",
    }

    with torch.no_grad():
        for name in sorted(missing):
            vision_marker = next(
                (marker for marker in vision_name_map if marker in name),
                None,
            )
            if vision_marker is not None and name in parameters:
                source_name = name.replace(
                    vision_marker,
                    vision_name_map[vision_marker],
                    1,
                )
                source = parameters.get(source_name)
                target = parameters[name]
                if source is None or source.shape != target.shape:
                    raise ValueError(
                        f"Cannot initialize missing {name} from {source_name}: "
                        "source is absent or has a different shape."
                    )
                target.copy_(source)
                handled.add(name)
                logger.info("Initialized missing %s from %s", name, source_name)
                continue

            if name.endswith(".rotary_emb.inv_freq") and name in buffers:
                module_name = name.rsplit(".inv_freq", 1)[0]
                module = dict(model.named_modules())[module_name]
                dim = module.config.d_model // module.config.n_heads
                target = buffers[name]
                inv_freq = 1.0 / (
                    module.rope_theta
                    ** (
                        torch.arange(0, dim, 2, device=target.device, dtype=torch.float)
                        / dim
                    )
                )
                target.copy_(inv_freq.to(dtype=target.dtype))
                handled.add(name)
                logger.info("Recomputed missing %s", name)

    unexpected = sorted(missing - handled)
    if unexpected:
        preview = "\n  ".join(unexpected[:50])
        suffix = "\n  ..." if len(unexpected) > 50 else ""
        raise ValueError(
            "Checkpoint is missing unsupported model keys:\n  "
            f"{preview}{suffix}"
        )


def validate_model_vocab(
    model: PreTrainedModel,
    expected_size: int,
) -> None:
    """Validate the actual embedding/output rows and their saved config fields."""
    input_rows = model.get_input_embeddings().weight.shape[0]
    if input_rows != expected_size:
        raise ValueError(
            f"Input embedding has {input_rows} rows; expected {expected_size}."
        )

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        output_rows = output_embeddings.weight.shape[0]
        if output_rows != expected_size:
            raise ValueError(
                f"Output projection has {output_rows} rows; expected {expected_size}."
            )

    for field in ("vocab_size", "embedding_size", "extended_vocab_size"):
        value = getattr(model.config, field, None)
        if value is not None and int(value) != expected_size:
            raise ValueError(
                f"Model config {field}={value}; expected {expected_size}."
            )


def load_tokenizer(
    model_or_tokenizer_path: str,
) -> "PreTrainedTokenizerFast":
    """
    加载 Gestalt-DiMOO 专用 tokenizer（含 16,384 个 VQVAE 视觉 token）。

    tokenizer 可寻址 ID 为 [0, 142740)，模型 embedding/output padding 到
    142848；padding 区间同时容纳训练时直接注入的 interaction token ID。

    Args:
        model_or_tokenizer_path: 包含 ``tokenizer.json`` 的模型 checkpoint
            目录；也允许直接传入 tokenizer.json，便于测试或显式覆盖。

    Returns:
        已配置 bos/eos/pad token 的 PreTrainedTokenizerFast 实例
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from .config import SPECIAL_TOKENS

    source = Path(model_or_tokenizer_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(
            f"Tokenizer/checkpoint path does not exist: {source}"
        )

    if source.is_file():
        if source.name != "tokenizer.json":
            raise ValueError(
                f"Tokenizer file must be named tokenizer.json, got {source.name!r}."
            )
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(source))
    elif (source / "tokenizer.json").is_file():
        # Loading the canonical file directly also works when a custom model_type
        # is not registered with AutoTokenizer.
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(source / "tokenizer.json")
        )
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(source),
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as error:
            raise FileNotFoundError(
                f"Checkpoint {source} does not contain a loadable tokenizer. "
                "Save tokenizer.json (or standard Hugging Face tokenizer files) "
                "inside the final checkpoint."
            ) from error

    logger.info("Loaded Gestalt tokenizer from %s", source)

    # 设置 special token 属性（tokenizer.json 裸加载时为 None）
    tokenizer.bos_token = "<|startoftext|>"
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = "<padding>"

    expected_ids = {
        "<|startoftext|>": SPECIAL_TOKENS.BOS,
        "<|endoftext|>": SPECIAL_TOKENS.EOS,
        "<system>": SPECIAL_TOKENS.SYSTEM_START,
        "</system>": SPECIAL_TOKENS.SYSTEM_END,
        "<user>": SPECIAL_TOKENS.USER_START,
        "</user>": SPECIAL_TOKENS.USER_END,
        "<|mdm_mask|>": SPECIAL_TOKENS.MASK,
        "<padding>": SPECIAL_TOKENS.PADDING,
        "<IMAGE>": SPECIAL_TOKENS.IMAGE_START,
        "</IMAGE>": SPECIAL_TOKENS.IMAGE_END,
        "<answer>": SPECIAL_TOKENS.ANSWER_START,
        "</answer>": SPECIAL_TOKENS.ANSWER_END,
    }
    mismatches = {
        token: (tokenizer.convert_tokens_to_ids(token), expected)
        for token, expected in expected_ids.items()
        if tokenizer.convert_tokens_to_ids(token) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{token}: actual={actual}, expected={expected}"
            for token, (actual, expected) in mismatches.items()
        )
        raise ValueError(f"Tokenizer special-token ID mismatch: {details}")

    expected_tokenizer_size = SPECIAL_TOKENS.VISUAL_TOKEN_END
    if len(tokenizer) != expected_tokenizer_size:
        raise ValueError(
            f"Tokenizer size is {len(tokenizer)}, expected {expected_tokenizer_size}."
        )

    logger.info(
        f"Tokenizer loaded: len={len(tokenizer)}, "
        f"bos_id={tokenizer.bos_token_id}, eos_id={tokenizer.eos_token_id}, "
        f"pad_id={tokenizer.pad_token_id}"
    )
    return tokenizer
