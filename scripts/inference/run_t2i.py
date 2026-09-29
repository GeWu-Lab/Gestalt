#!/usr/bin/env python3
"""
Gestalt Text-to-Image Inference (Stage3 template)

Generates VQVAE visual tokens from a text prompt using the Stage3 SFT
conversation template and the MaskGit-style parallel sampling decoder
(generate_t2i).

Usage:
    python scripts/inference/run_t2i.py \
        --model /path/to/checkpoint \
        --prompt "A cat sitting on a windowsill" \
        --output output_tokens.npy \
        --lat-h 32 --lat-w 32 \
        --steps 64 --temperature 1.0 --cfg-scale 4.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gestalt.model.modeling_gestalt import GestaltModelLM
from gestalt.model.builder import load_tokenizer, validate_model_vocab
from gestalt.model.config import VOCAB_CONFIG


def main():
    parser = argparse.ArgumentParser(
        description="Gestalt T2I inference (Text -> Image tokens)"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to a Gestalt checkpoint directory",
    )
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for image generation")
    parser.add_argument("--output", type=str, default="output_tokens.npy",
                        help="Output path for the generated VQVAE codebook "
                             "indices (.npy)")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="System prompt used during training (default: empty)")
    parser.add_argument("--lat-h", type=int, default=32,
                        help="Latent image token grid height")
    parser.add_argument("--lat-w", type=int, default=32,
                        help="Latent image token grid width")
    parser.add_argument("--steps", type=int, default=64,
                        help="Number of MaskGit sampling steps")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Gumbel-Max sampling temperature")
    parser.add_argument("--cfg-scale", type=float, default=4.0,
                        help="Classifier-Free Guidance scale (0.0 = disabled)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model)

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

    system_prompt_text = args.system_prompt or ""
    system_prompt_tokens = tokenizer(
        system_prompt_text, add_special_tokens=False
    )["input_ids"]

    user_prompt_tokens = tokenizer(
        args.prompt, add_special_tokens=False
    )["input_ids"]

    print(f"Generating {args.lat_h}x{args.lat_w} image tokens "
          f"(steps={args.steps}, temp={args.temperature}, "
          f"cfg={args.cfg_scale}) ...")
    with torch.no_grad():
        visual_tokens = model.generate_t2i(
            system_prompt_tokens=system_prompt_tokens,
            user_prompt_tokens=user_prompt_tokens,
            lat_h=args.lat_h,
            lat_w=args.lat_w,
            timesteps=args.steps,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
        )

    raw_codes = (visual_tokens[0].cpu().numpy()
                 - VOCAB_CONFIG.VISUAL_TOKEN_OFFSET).astype(np.int32)

    if np.any(raw_codes < 0) or np.any(raw_codes >= VOCAB_CONFIG.VQVAE_CODEBOOK_SIZE):
        raise RuntimeError("Model returned out-of-range VQVAE codebook indices.")

    np.save(args.output, raw_codes)
    print(f"Saved {raw_codes.shape[0]} codebook indices to {args.output}")
    print(f"  shape : {raw_codes.shape}")
    print(f"  range : [{raw_codes.min()}, {raw_codes.max()}]")


if __name__ == "__main__":
    main()
