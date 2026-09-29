#!/usr/bin/env python3
"""
Stage-III SFT Script

Usage:
    torchrun --nproc_per_node=N scripts/training/train_stage3.py \
        --pretrained /path/to/pretrained/checkpoint \
        --data /mnt/shared-storage-gpfs2/ocvlm/data_arrangement/stage3_filtered \
        --config configs/training/gestalt_stage_3.yaml \
        --deepspeed configs/deepspeed/zero_stage2.json \
        --output data/outputs/stage3
"""

import argparse
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gestalt.data.datasets import Stage3Dataset, TokenBudgetPackingDataset
from gestalt.data.stage3 import create_stage3_processor
from gestalt.model.builder import load_tokenizer
from gestalt.model.config import SPECIAL_TOKENS, VOCAB_CONFIG
from gestalt.training.trainer import create_trainer


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_config(cfg: dict) -> None:
    expected_vocab = {
        "base_vocab_size": VOCAB_CONFIG.BASE_VOCAB_SIZE,
        "vqvae_codebook_size": VOCAB_CONFIG.VQVAE_CODEBOOK_SIZE,
        "extended_vocab_size": VOCAB_CONFIG.EXTENDED_VOCAB_SIZE,
    }
    for key, expected in expected_vocab.items():
        if cfg.get(key) != expected:
            raise ValueError(f"{key} must be {expected}, got {cfg.get(key)!r}.")
    for key in ("max_length", "token_budget", "inter_layer_num"):
        if not isinstance(cfg.get(key), int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    if cfg["token_budget"] < cfg["max_length"]:
        raise ValueError("token_budget must be greater than or equal to max_length.")
    if SPECIAL_TOKENS.INTERACTION_TOKEN_END > cfg["extended_vocab_size"]:
        raise ValueError("extended_vocab_size does not contain all interaction tokens.")


def main():
    parser = argparse.ArgumentParser(description="Stage-III SFT Training")
    parser.add_argument("--config", type=str, default="configs/training/gestalt_stage_3.yaml")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/outputs/stage3")
    parser.add_argument("--pretrained", type=str, required=True,
                        help="Path to pretrained Gestalt/LLaDA checkpoint")
    parser.add_argument("--deepspeed", type=str, default="configs/deepspeed/zero_stage2.json",
                        help="DeepSpeed config path. Use 'none' to disable.")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max_steps from config")
    parser.add_argument("--logging_steps", type=int, default=None, help="Override logging_steps from config")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.logging_steps is not None:
        cfg["logging_steps"] = args.logging_steps
    validate_config(cfg)
    print(f"Loaded config from {args.config}")
    
    print(f"deepspeed: {args.deepspeed}")
    print(f"gradient_checkpointing: {cfg.get('gradient_checkpointing', True)}")

    tokenizer = load_tokenizer(args.pretrained)
    print(f"Loaded Gestalt-DiMOO tokenizer, len={len(tokenizer)}")
    

    # Processor: masking only, no tokenization
    processor = create_stage3_processor(
        tokenizer=tokenizer,
        max_length=cfg.get("max_length", 5120),
        answer_mask_schedule=cfg.get("answer_mask_schedule", "cosine"),
        image_mask_schedule=cfg.get("image_mask_schedule", "linear"),
        image_loss_weight=cfg.get("image_loss_weight", 1.0),
        loss_weight_mode=cfg.get("loss_weight_mode", "sample"),
    )

    # Dataset: streaming IterableDataset (no upfront indexing)
    raw_dataset = Stage3Dataset(args.data, processor)
    print(f"Loaded dataset with {len(raw_dataset)} examples")
    token_budget = cfg.get("token_budget", 10000)
    dataset = TokenBudgetPackingDataset(raw_dataset, max_length=token_budget)
    print(f"Wrapped with TokenBudgetPackingDataset (budget={token_budget})")

    max_steps = args.max_steps if args.max_steps is not None else cfg.get("max_steps")
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer for the streaming dataset.")

    # Validate vocab config
    print(f"Vocab config - base: {cfg['base_vocab_size']}, "
          f"vqvae: {cfg['vqvae_codebook_size']}, "
          f"extended: {cfg['extended_vocab_size']}")
    
    trainer = create_trainer(
        stage="stage3",
        model_name_or_path=args.pretrained,
        train_dataset=dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        output_dir=args.output,
        # Optimizer (Technical Report)
        learning_rate=cfg.get("learning_rate", 1e-5),
        weight_decay=cfg.get("weight_decay", 0.1),
        adam_beta1=cfg.get("adam_beta1", 0.9),
        adam_beta2=cfg.get("adam_beta2", 0.95),
        adam_epsilon=cfg.get("adam_epsilon", 1e-8),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        # LR schedule
        lr_scheduler_type=cfg.get("lr_scheduler_type", "constant"),
        warmup_steps=cfg.get("warmup_steps", 100),
        # Batch
        # Note: TokenBudgetPackingDataset requires per_device_train_batch_size=1
        # The packing is done within the dataset
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        # Training duration
        max_steps=max_steps,
        # Logging & saving
        save_steps=cfg.get("save_steps", 400),
        save_total_limit=cfg.get("save_total_limit", 5),
        logging_steps=cfg.get("logging_steps", 50),
        # Sequence length (must match processor's max_length)
        max_length=cfg.get("max_length", 5120),
        # Misc
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 0),
        remove_unused_columns=False,
        # Stage-3 specific
        vocab_size=cfg.get("extended_vocab_size", 142848),
        # DeepSpeed
        deepspeed=args.deepspeed if args.deepspeed.lower() != "none" else None,
        inter_layer_state=cfg.get("inter_layer_state", "interaction_token"),
        inter_layer_num=cfg.get("inter_layer_num", 8),
    )

    print(f"Starting Stage-III training → {args.output}")
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Training complete; final model and tokenizer saved to {args.output}")


if __name__ == "__main__":
    main()
