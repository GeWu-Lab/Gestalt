"""Model utilities used by Stage-III SFT training."""

import importlib
import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import PreTrainedModel


logger = logging.getLogger(__name__)


def extend_llada_vocab(
    model: PreTrainedModel,
    target_vocab_size: int,
    new_embedding_init: str = "mean",
) -> None:
    """Resize LLaDA input/output vocabulary while preserving existing rows."""
    wte = model.get_input_embeddings()
    current_size = wte.weight.shape[0]

    if current_size >= target_vocab_size:
        logger.info(
            "Vocabulary already satisfies target size: current=%s, target=%s",
            current_size,
            target_vocab_size,
        )
        return

    logger.info("Extending vocabulary from %s to %s", current_size, target_vocab_size)

    d_model = wte.weight.shape[1]
    device = wte.weight.device
    dtype = wte.weight.dtype

    new_wte = nn.Embedding(target_vocab_size, d_model, device=device, dtype=dtype)

    with torch.no_grad():
        new_wte.weight.data[:current_size] = wte.weight.data[:current_size]

        num_new = target_vocab_size - current_size
        if new_embedding_init == "mean":
            mean_emb = wte.weight.data.mean(dim=0, keepdim=True)
            new_wte.weight.data[current_size:] = mean_emb.expand(num_new, -1)
        else:
            new_wte.weight.data[current_size:].zero_()

    model.set_input_embeddings(new_wte)

    if not (hasattr(model, "model")
            and hasattr(model.model, "transformer")
            and hasattr(model.model.transformer, "ff_out")):
        raise AttributeError(
            "Expected LLaDA output projection at model.model.transformer.ff_out."
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

    model.config.vocab_size = target_vocab_size
    # LLaDA persists embedding_size separately from vocab_size.
    if hasattr(model.config, "embedding_size"):
        model.config.embedding_size = target_vocab_size

    logger.info(
        "Vocabulary extension complete: wte=%s, ff_out=%s",
        model.get_input_embeddings().weight.shape,
        model.model.transformer.ff_out.weight.shape,
    )


def enable_llada_gradient_checkpointing(model: PreTrainedModel) -> None:
    """Install the Hugging Face checkpointing hook for a base LLaDA model."""
    cfg_modname = model.config.__class__.__module__
    mdl_modname = model.__class__.__module__

    cfg_mod = importlib.import_module(cfg_modname)
    mdl_mod = importlib.import_module(mdl_modname)

    ActivationCheckpointingStrategy = cfg_mod.ActivationCheckpointingStrategy

    logger.info(
        "Installing gradient-checkpointing hook: config=%s, model=%s, strategy=%s",
        cfg_modname,
        mdl_modname,
        ActivationCheckpointingStrategy,
    )

    def _gc_enable(self, gradient_checkpointing_kwargs=None):
        self.config.use_cache = False
        self.model.set_activation_checkpointing(
            ActivationCheckpointingStrategy.whole_layer
        )

    setattr(mdl_mod.LLaDAModelLM, "supports_gradient_checkpointing", True)
    setattr(mdl_mod.LLaDAModelLM, "gradient_checkpointing_enable", _gc_enable)

    logger.info("Gradient-checkpointing hook installed with whole_layer strategy.")


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
    checkpoint_path: str,
) -> "PreTrainedTokenizerFast":
    """Load and validate the tokenizer stored in a model checkpoint."""
    from transformers import PreTrainedTokenizerFast
    from .config import SPECIAL_TOKENS

    source = Path(checkpoint_path).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {source}")
    tokenizer_file = source / "tokenizer.json"
    if not tokenizer_file.is_file():
        raise FileNotFoundError(
            f"Checkpoint {source} does not contain tokenizer.json."
        )
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))

    logger.info("Loaded Gestalt tokenizer from %s", source)

    # Raw tokenizer.json files do not expose these attributes automatically.
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
