#!/usr/bin/env python3
"""
Gestalt-DiMOO MMU (Multimodal Understanding) Inference

Image-to-Text: given pre-extracted VQVAE image tokens and a text question,
generate a text answer using the discrete-diffusion decoder.

Usage:
    python scripts/inference/run_mmu.py \
        --model /path/to/checkpoint \
        --image-tokens /path/to/image_tokens.npy \
        --question "What is in this image?" \
        --token-h 32 --token-w 32 \
        --steps 64 --gen-length 128 --temperature 0.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure the gestalt package is importable (same pattern as train_stage3.py)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gestalt.model.modeling_gestalt import GestaltModelLM
from gestalt.model.builder import load_tokenizer, validate_model_vocab
from gestalt.model.config import VOCAB_CONFIG
from gestalt.model.templates import DEFAULT_MMU_SYSTEM_PROMPT


def main():
    parser = argparse.ArgumentParser(
        description="Gestalt-DiMOO MMU inference (Image+Question -> Text)"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to a Gestalt checkpoint directory",
    )
    parser.add_argument(
        "--image-tokens", type=str, required=True,
        help="Path to a .npy file containing raw VQVAE codebook indices "
             "(shape: [H*W] or [H, W], dtype int)",
    )
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--token-h", type=int, default=32,
                        help="Image token grid height")
    parser.add_argument("--token-w", type=int, default=32,
                        help="Image token grid width")
    parser.add_argument("--gen-length", type=int, default=128,
                        help="Maximum number of answer tokens to generate")
    parser.add_argument("--steps", type=int, default=64,
                        help="Number of diffusion decoding steps")
    parser.add_argument("--block-length", type=int, default=64,
                        help="Block length for block-wise decoding")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Gumbel sampling temperature (0.0 = argmax)")
    parser.add_argument("--remasking", type=str, default="low_confidence",
                        choices=["low_confidence", "random"],
                        help="Remasking strategy")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load tokenizer
    # ------------------------------------------------------------------
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model)

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    print(f"Loading model from {args.model} ...")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.device.startswith("cuda") and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The requested CUDA device does not support BF16 inference.")
    model_dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = GestaltModelLM.from_pretrained(
        args.model,
        dtype=model_dtype,
        device_map=None,
        trust_remote_code=False,
    )
    model = model.to(args.device)
    model.eval()
    validate_model_vocab(model, VOCAB_CONFIG.EXTENDED_VOCAB_SIZE)
    print(f"Model loaded (vocab_size={model.config.vocab_size})")

    # ------------------------------------------------------------------
    # 3. Load image tokens
    # ------------------------------------------------------------------
    raw_codes = np.load(args.image_tokens)
    if not np.issubdtype(raw_codes.dtype, np.integer):
        raise ValueError(
            f"Image tokens must have an integer dtype, got {raw_codes.dtype}."
        )
    raw_codes = raw_codes.flatten().astype(np.int64, copy=False)
    expected_len = args.token_h * args.token_w
    if raw_codes.shape[0] != expected_len:
        raise ValueError(
            f"Image token count ({raw_codes.shape[0]}) must equal "
            f"token_h*token_w ({expected_len})."
        )
    if np.any(raw_codes < 0) or np.any(raw_codes >= VOCAB_CONFIG.VQVAE_CODEBOOK_SIZE):
        raise ValueError(
            "Image tokens must be raw VQVAE codebook indices in "
            f"[0, {VOCAB_CONFIG.VQVAE_CODEBOOK_SIZE})."
        )

    # Convert raw codebook indices -> model token IDs
    image_token_ids = (raw_codes + VOCAB_CONFIG.VISUAL_TOKEN_OFFSET).tolist()

    # ------------------------------------------------------------------
    # 4. Tokenize the question
    # ------------------------------------------------------------------
    question_tokens = tokenizer(
        args.question, add_special_tokens=False
    )["input_ids"]

    # ------------------------------------------------------------------
    # 5. Optionally build a system prompt (use the Stage3 default)
    # ------------------------------------------------------------------
    system_prompt_text = DEFAULT_MMU_SYSTEM_PROMPT
    system_tokens = tokenizer(
        system_prompt_text, add_special_tokens=False
    )["input_ids"]

    # ------------------------------------------------------------------
    # 6. Generate
    # ------------------------------------------------------------------
    print("Generating...")
    with torch.no_grad():
        answer_token_ids = model.generate_mmu(
            image_tokens=image_token_ids,
            token_h=args.token_h,
            token_w=args.token_w,
            system_tokens=system_tokens,
            question_tokens=question_tokens,
            gen_length=args.gen_length,
            steps=args.steps,
            block_length=args.block_length,
            temperature=args.temperature,
            remasking=args.remasking,
        )

    # ------------------------------------------------------------------
    # 7. Decode and print
    # ------------------------------------------------------------------
    answer_text = tokenizer.decode(answer_token_ids, skip_special_tokens=True)
    print("\n" + "=" * 60)
    print(f"Question : {args.question}")
    print(f"Answer   : {answer_text}")
    print("=" * 60)


if __name__ == "__main__":
    main()
