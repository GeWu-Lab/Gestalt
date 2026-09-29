"""Hugging Face Trainer integration for Stage-III SFT."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.trainer_utils import EvalPrediction

from gestalt.model.config import VOCAB_CONFIG
from gestalt.model.modeling_gestalt import GestaltModelLM

logger = logging.getLogger(__name__)

@dataclass
class GestaltTrainingConfig(TrainingArguments):
    """Training arguments required by Gestalt Stage-III SFT."""

    model_name_or_path: str = None
    gradient_checkpointing: bool = True
    base_model: str = "Alpha-VLLM/LLaDA-8B-Instruct"
    vocab_size: int = 142848

    stage: str = "stage3"

    save_only_model: bool = False
    min_lr_rate: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.stage != "stage3":
            raise ValueError(f"Only stage3 training is supported, got stage={self.stage!r}")


class GestaltTrainer(Trainer):
    """Trainer with globally normalized weighted CE and pre-sharded data."""

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        args: GestaltTrainingConfig,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        data_collator: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        **kwargs,
    ):
        import inspect as _inspect
        _trainer_params = set(_inspect.signature(Trainer.__init__).parameters)
        # Transformers 4.46 renamed tokenizer to processing_class.
        if "processing_class" in _trainer_params and "tokenizer" not in _trainer_params:
            super().__init__(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                **kwargs,
            )
        else:
            super().__init__(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                **kwargs,
            )
        self.config = args
        # compute_loss performs its own global weighted normalization and does
        # not use Trainer's num_items_in_batch contract.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute globally normalized token-weighted cross entropy."""
        labels = inputs.pop("labels")
        input_ids = inputs["input_ids"]
        position_ids = inputs.get("position_ids")
        document_ids = inputs.get("document_ids")
        ce_loss_weights = inputs.pop("ce_loss_weights")
        inputs.pop("loss_mask", None)

        outputs, _, _ = model(
            input_ids=input_ids,
            position_ids=position_ids,
            document_ids=document_ids,
        )
            
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        vocab_size = logits.shape[-1]
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        per_token_loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100, reduction="none")

        valid_mask = labels_flat != -100
        weights_flat = ce_loss_weights.reshape(-1).float()
        if not torch.isfinite(weights_flat).all() or (weights_flat < 0).any():
            raise ValueError("ce_loss_weights must be finite and non-negative.")
        if (weights_flat[~valid_mask] != 0).any():
            raise ValueError("Ignored labels must have zero CE loss weight.")
        if (weights_flat[valid_mask] <= 0).any():
            raise ValueError("Every supervised label must have a positive CE loss weight.")

        def _global_sum(t: torch.Tensor) -> torch.Tensor:
            t = t.detach().clone()
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            return t

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1

        weighted_loss = per_token_loss * weights_flat
        local_weight_sum = weights_flat.sum()
        local_weighted_loss_sum = weighted_loss.sum()
        global_weight_sum = _global_sum(local_weight_sum)
        if global_weight_sum.item() <= 0:
            raise ValueError("The batch has no positive supervised loss weight.")
        loss = local_weighted_loss_sum * world_size / global_weight_sum

        if return_outputs:
            return loss, outputs
        return loss

    def get_train_dataloader(self):
        """Build train DataLoader without Accelerate iterable sharding.

        GestaltDataset already shards by rank and worker; Accelerate must not
        wrap it in IterableDatasetShard a second time.
        """
        import torch.utils.data as tud

        if not isinstance(self.train_dataset, tud.IterableDataset):
            return super().get_train_dataloader()

        data_collator = self._get_collator_with_removed_columns(
            self.data_collator,
            description="training",
        )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if self.args.dataloader_num_workers > 0:
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return tud.DataLoader(self.train_dataset, **dataloader_params)

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Build a LambdaLR after normalizing DeepSpeed learning-rate values."""
        from functools import partial
        from torch.optim.lr_scheduler import LambdaLR

        _opt = optimizer or self.optimizer
        if _opt is None:
            return super().create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

        # DeepSpeed may expose learning rates as strings, lists, or tensors.
        try:
            for pg in _opt.param_groups:
                for key in ("lr", "initial_lr"):
                    v = pg.get(key)
                    if v is None:
                        continue
                    if isinstance(v, (list, tuple)) and len(v) >= 1:
                        pg[key] = float(v[0])
                    elif hasattr(v, "item"):
                        pg[key] = float(v.item())
                    elif isinstance(v, str):
                        pg[key] = float(v)
        except (AttributeError, TypeError):
            pass

        try:
            n_groups = len(_opt.param_groups)
        except (AttributeError, TypeError):
            n_groups = 1

        scheduler_type = getattr(self.args, "lr_scheduler_type", "linear")
        scheduler_type_str = scheduler_type.value if hasattr(scheduler_type, "value") else str(scheduler_type)

        if scheduler_type_str in ("constant", "constant_with_warmup"):
            num_warmup_steps = self.args.get_warmup_steps(num_training_steps)

            def _constant_lr_lambda(current_step: int, *, num_warmup: int) -> float:
                if current_step < num_warmup:
                    return float(current_step) / float(max(1, num_warmup))
                return 1.0

            lr_lambda = partial(_constant_lr_lambda, num_warmup=num_warmup_steps)
            self.lr_scheduler = LambdaLR(_opt, [lr_lambda] * n_groups, last_epoch=-1)

        elif scheduler_type_str == "cosine":
            from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
            num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
            min_lr_rate = getattr(self.args, "min_lr_rate", 0.5)
            self.lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
                _opt,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                min_lr_rate=min_lr_rate,
            )

        else:
            num_warmup_steps = self.args.get_warmup_steps(num_training_steps)

            def _linear_lr_lambda(current_step: int, *, num_warmup: int, num_total: int) -> float:
                if current_step < num_warmup:
                    return float(current_step) / float(max(1, num_warmup))
                return max(0.0, float(num_total - current_step) / float(max(1, num_total - num_warmup)))

            lr_lambda = partial(
                _linear_lr_lambda,
                num_warmup=num_warmup_steps,
                num_total=num_training_steps,
            )
            self.lr_scheduler = LambdaLR(_opt, [lr_lambda] * n_groups, last_epoch=-1)

        return self.lr_scheduler

def _load_checkpoint_tensor(checkpoint_path: str, tensor_name: str) -> Optional[torch.Tensor]:
    """Read one tensor from a safetensors checkpoint without loading the full model."""
    def _read_safetensors_tensor(path: Path, name: str) -> Optional[torch.Tensor]:
        dtype_map = {
            "BF16": torch.bfloat16,
            "F16": torch.float16,
            "F32": torch.float32,
            "F64": torch.float64,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }

        with open(path, "rb") as f:
            header_size = int.from_bytes(f.read(8), byteorder="little", signed=False)
            header = json.loads(f.read(header_size))
            metadata = header.get(name)
            if metadata is None:
                return None

            dtype = dtype_map.get(metadata["dtype"])
            if dtype is None:
                raise ValueError(f"Unsupported safetensors dtype {metadata['dtype']} for {name}")

            begin, end = metadata["data_offsets"]
            f.seek(8 + header_size + begin)
            raw = bytearray(f.read(end - begin))

        tensor = torch.frombuffer(raw, dtype=dtype)
        return tensor.reshape(metadata["shape"]).clone()

    path = Path(checkpoint_path)
    if path.is_file():
        candidates = [path]
    else:
        index_path = path / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            shard = index.get("weight_map", {}).get(tensor_name)
            candidates = [path / shard] if shard else []
        else:
            candidates = [
                path / "model.safetensors",
                path / "pytorch_model.safetensors",
            ]

    for candidate in candidates:
        if candidate is None or not candidate.exists() or candidate.suffix != ".safetensors":
            continue
        tensor = _read_safetensors_tensor(candidate, tensor_name)
        if tensor is not None:
            return tensor

    return None


def _fold_interaction_adapter_into_wte(
    model: PreTrainedModel,
    loading_info: Dict[str, Any],
    checkpoint_path: str,
) -> bool:
    """
    Convert Midtrain's interaction_embedding_adapter into normal SFT token embeddings.

    Midtrain Stage-II trained a separate `interaction_embedding_adapter.weight`.
    Stage-III SFT should use regular token IDs, so copy that adapter into
    `model.transformer.wte[INTERACTION_START:INTERACTION_END]` and then disable
    the runtime adapter module. If the checkpoint has no adapter, this is a no-op.
    """
    tensor_name = "interaction_embedding_adapter.weight"
    missing_keys = set(loading_info.get("missing_keys", []))
    adapter = getattr(model, "interaction_embedding_adapter", None)

    adapter_weight = None
    if adapter is not None and tensor_name not in missing_keys:
        adapter_weight = adapter.weight.detach()
        logger.info("Folding loaded %s into interaction token embeddings.", tensor_name)
    else:
        adapter_weight = _load_checkpoint_tensor(checkpoint_path, tensor_name)
        if adapter_weight is not None:
            logger.info("Folding checkpoint %s into interaction token embeddings.", tensor_name)

    if adapter_weight is None:
        if tensor_name in set(loading_info.get("unexpected_keys", [])):
            raise ValueError(
                f"Checkpoint contains {tensor_name}, but it could not be read for "
                "folding. Use a safetensors checkpoint or save the adapter with "
                "use_interaction_embedding_adapter=true in config.json."
            )
        if adapter is not None:
            delattr(model, "interaction_embedding_adapter")
        if hasattr(model.config, "use_interaction_embedding_adapter"):
            model.config.use_interaction_embedding_adapter = False
        print("--- No interaction adapter found in checkpoint; skipping adapter folding ---")
        return False

    from ..model.config import SPECIAL_TOKENS as ModelSpecialTokens

    start = ModelSpecialTokens.INTERACTION_TOKEN_START
    end = ModelSpecialTokens.INTERACTION_TOKEN_END
    expected_rows = end - start
    if adapter_weight.shape[0] != expected_rows:
        raise ValueError(
            f"{tensor_name} has {tuple(adapter_weight.shape)}, expected "
            f"({expected_rows}, hidden_size) for interaction token range [{start}, {end})."
        )

    wte = model.get_input_embeddings()
    if wte.weight.shape[0] < end:
        raise ValueError(
            f"Input embedding vocab size {wte.weight.shape[0]} is smaller than "
            f"interaction token end id {end}."
        )
    if wte.weight.shape[1] != adapter_weight.shape[1]:
        raise ValueError(
            f"{tensor_name} hidden size {adapter_weight.shape[1]} does not match "
            f"input embedding hidden size {wte.weight.shape[1]}."
        )

    with torch.no_grad():
        wte.weight.data[start:end].copy_(
            adapter_weight.to(device=wte.weight.device, dtype=wte.weight.dtype)
        )

    if adapter is not None:
        delattr(model, "interaction_embedding_adapter")
    if hasattr(model.config, "use_interaction_embedding_adapter"):
        model.config.use_interaction_embedding_adapter = False

    print(
        f"--- Folded interaction adapter into wte[{start}:{end}] "
        f"with shape {tuple(adapter_weight.shape)} ---"
    )
    return True


def create_trainer(
    stage: str,
    model_name_or_path: str,
    train_dataset,
    eval_dataset: Optional[Any] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    inter_layer_state: str = "interaction_token",
    **kwargs,
) -> GestaltTrainer:
    """Load the model and construct the Stage-III trainer."""
    from ..data.datasets import DataCollatorForCausalLM

    vocab_size = kwargs.get("vocab_size", VOCAB_CONFIG.EXTENDED_VOCAB_SIZE)
    inter_layer_num = kwargs.pop("inter_layer_num", None)

    if stage != "stage3":
        raise ValueError(f"Only stage3 training is supported, got stage={stage!r}")
    if inter_layer_state != "interaction_token":
        raise ValueError(
            "Stage3 only supports inter_layer_state='interaction_token', "
            f"got {inter_layer_state!r}."
        )

    print("model_name_or_path is ", model_name_or_path)
    use_bf16 = bool(kwargs.get("bf16", False))
    if use_bf16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16=true, but the current CUDA device does not support BF16.")
    from_pretrained_kwargs = dict(
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map=None,
        trust_remote_code=False,
        output_loading_info=True,
        inter_layer_state=inter_layer_state,
    )
    if inter_layer_num is not None:
        from_pretrained_kwargs["inter_layer_num"] = inter_layer_num
    model, loading_info = GestaltModelLM.from_pretrained(
        model_name_or_path,
        **from_pretrained_kwargs,
    )
    print(
        f"--- Model loaded with inter_layer_num="
        f"outer_config={model.config.inter_layer_num}, "
        f"inner_model={model.model.inter_layer_num} "
        f"(requested={inter_layer_num}) ---"
    )

    unexpected_keys = [
        key for key in loading_info.get("unexpected_keys", [])
        if key != "interaction_embedding_adapter.weight"
    ]
    mismatched_keys = loading_info.get("mismatched_keys", [])
    if unexpected_keys or mismatched_keys:
        raise ValueError(
            "Checkpoint is incompatible with the Gestalt model: "
            f"unexpected_keys={unexpected_keys}, mismatched_keys={mismatched_keys}"
        )

    missing_keys = loading_info.get("missing_keys", [])
    structural_missing = [
        key for key in missing_keys
        if key != "interaction_embedding_adapter.weight"
    ]
    print("--- Checking missing parameters and vocab size ---")
    if structural_missing:
        print(
            "--- Missing checkpoint parameters detected; "
            "initializing newly introduced parameters ---"
        )
        from ..model.builder import _reinit_missing_parameters
        _reinit_missing_parameters(model, structural_missing)
    else:
        print("--- No structural missing parameters detected; preserving checkpoint weights ---")

    loaded_vocab_size = model.get_input_embeddings().weight.shape[0]
    if loaded_vocab_size < vocab_size:
        model.resize_and_initialize_vocab(vocab_size)
    elif loaded_vocab_size == vocab_size:
        print(f"--- Vocab size already {loaded_vocab_size}; skipping resize ---")
    else:
        raise ValueError(
            f"Checkpoint vocab size {loaded_vocab_size} exceeds configured size {vocab_size}."
        )

    _fold_interaction_adapter_into_wte(model, loading_info, model_name_or_path)
    from ..model.builder import validate_model_vocab
    validate_model_vocab(model, vocab_size)
    
    if kwargs.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    
    if tokenizer is None:
        from ..model.builder import load_tokenizer as _load_tokenizer
        tokenizer = _load_tokenizer(model_name_or_path)

    kwargs.pop("max_length", None)
    data_collator = DataCollatorForCausalLM()
    
    config = GestaltTrainingConfig(
        model_name_or_path=model_name_or_path,
        stage=stage,
        report_to="none",
        **kwargs,
    )

    trainer = GestaltTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    return trainer
