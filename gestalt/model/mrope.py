"""
MRoPE2DRotaryEmbedding: 2D Multi-Resolution Rotary Position Embedding.

Replaces standard 1D RoPE with 2D positional encoding that assigns separate
height and width position IDs to image tokens while preserving identical
behavior for text tokens (where h_pos == w_pos).

Each half is then duplicated via cat([freqs, freqs]) to match the interleaved
rotate_half convention, yielding layout [h, w, h, w] across the full 128 dims.

When h_pos == w_pos (text tokens), this reduces to exactly the original 1D RoPE.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import SPECIAL_TOKENS, VOCAB_CONFIG


class MRoPE2DRotaryEmbedding(nn.Module):
    """2D Rotary Position Embedding for multimodal discrete diffusion.

    Args:
        config: Model config with attrs d_model, n_heads, rope_theta, rope_full_precision.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rope_theta = config.rope_theta

        dim = config.d_model // config.n_heads  # head_dim = 128
        # Full inv_freq with 64 values, identical to original RotaryEmbedding
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)  # shape (64,)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Interleaved rotate_half, identical to original RotaryEmbedding."""
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(
        self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)
    
    def _compute_freqs(
        self, position_ids: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute sin/cos embeddings from 2D position IDs.

        Args:
            position_ids: (2, B, T) where [0]=height, [1]=width positions.
            device: Target device.

        Returns:
            (pos_sin, pos_cos) each of shape (B, 1, T, head_dim).
        """
        # Reuse the registered buffer from __init__ (already correct dtype/values).
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float)

        # Interleaved frequency split:
        #   inv_freq_h gets the even-indexed inv_freq values (0, 2, 4, ..., 62)
        #   inv_freq_w gets the odd-indexed  inv_freq values (1, 3, 5, ..., 63)
        # Both halves therefore span the FULL frequency range (high to low),
        inv_freq_h = inv_freq[0::2]  # (32,) — original indices 0, 2, 4, ..., 62
        inv_freq_w = inv_freq[1::2]  # (32,) — original indices 1, 3, 5, ..., 63

        h_pos = position_ids[0].float()  # (B, T)
        w_pos = position_ids[1].float()  # (B, T)

        # Compute frequency tensors: (B, T, 32) each
        freqs_h = h_pos.unsqueeze(-1) * inv_freq_h
        freqs_w = w_pos.unsqueeze(-1) * inv_freq_w

        # Interleaved output layout: [h0, w0, h1, w1, ..., h31, w31], then
        # repeated to fill the second half of head_dim (so that rotate_half
        # pairs (i, i+64) reference the same frequency).
        #
        # Critical property: when h_pos == w_pos (text tokens), this collapses
        # exactly to the original 1D RoPE layout cat([p*inv_freq, p*inv_freq]),
        # so Dream's pretrained text RoPE semantics are preserved bit-for-bit.
        # When h_pos != w_pos (image tokens), even dims encode h via the
        # even-index frequencies and odd dims encode w via the odd-index
        # frequencies — both axes now have full-spectrum resolution.
        #
        # Do NOT change this to a block-concat like cat([h, w, h, w]); doing so
        # shuffles the dim->frequency mapping for text and breaks the Dream
        # pretrained text attention.
        interleaved = torch.stack([freqs_h, freqs_w], dim=-1).flatten(-2)  # (B, T, 64)
        positions = torch.cat([interleaved, interleaved], dim=-1)          # (B, T, 128)

        pos_sin = positions.sin().unsqueeze(1)  # (B, 1, T, 128)
        pos_cos = positions.cos().unsqueeze(1)  # (B, 1, T, 128)

        return pos_sin, pos_cos

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 2D rotary embeddings to queries and keys.

        Args:
            q: (B, n_heads, T, head_dim)
            k: (B, n_kv_heads, T, head_dim)
            position_ids: (2, B, T) — [0]=height, [1]=width

        Returns:
            (q_rotated, k_rotated) with same shapes as input.
        """
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            pos_sin, pos_cos = self._compute_freqs(position_ids, q.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)

            q_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, q_)
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)

        return q_.type_as(q), k_.type_as(k)


def compute_2d_position_ids(
    input_ids: torch.Tensor,
    image_grids: Optional[list] = None,
    document_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute 2D (height, width) position IDs for a token sequence.

    Text tokens get h_pos == w_pos (monotonically increasing). Image tokens
    get 2D grid positions based on their image_grid (H, W) dimensions.

    Args:
        input_ids: (B, L) token IDs.
        image_grids: List of (H, W) tuples for each image in the batch element.
            Consumed in order as image regions are encountered.
        document_ids: Optional ``(B, L)`` packed-document IDs. Position IDs
            restart at zero whenever the document ID changes.

    Returns:
        (2, B, L) tensor where [0] = height positions, [1] = width positions.

    Implementation strategy:
        Instead of a per-token Python loop (O(L) Python iterations), this
        function uses a segment-based approach:

        1. Locate all IMAGE_START / IMAGE_END positions with np.where to
           identify image regions vs text regions in a single scan.
        2. For each text segment, assign h_pos = w_pos = counter + arange(n)
           via a single numpy slice write — no per-token branching.
        3. For each image segment, compute row/col indices with vectorized
           divmod on np.arange(n_tokens) and write the whole region at once.

        The outer batch loop (B) is kept since B is typically 1 (document
        packing). The inner per-token loop is eliminated entirely.
    """
    if image_grids is None:
        image_grids = []
    B, L = input_ids.shape

    if document_ids is not None and document_ids.shape != input_ids.shape:
        raise ValueError(
            f"document_ids shape {tuple(document_ids.shape)} does not match "
            f"input_ids shape {tuple(input_ids.shape)}"
        )

    # One-shot copies to numpy avoid per-element CUDA synchronization.
    input_np = input_ids.detach().cpu().numpy()
    document_np = (
        document_ids.detach().cpu().numpy()
        if document_ids is not None
        else np.zeros((B, L), dtype=np.int64)
    )
    h_pos_np = np.zeros((B, L), dtype=np.int64)
    w_pos_np = np.zeros((B, L), dtype=np.int64)

    image_start_id = SPECIAL_TOKENS.IMAGE_START
    image_end_id = SPECIAL_TOKENS.IMAGE_END
    visual_offset = VOCAB_CONFIG.VISUAL_TOKEN_OFFSET
    visual_end = VOCAB_CONFIG.VISUAL_TOKEN_END
    mask_token_id = SPECIAL_TOKENS.MASK

    def _is_grid(value) -> bool:
        return (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(isinstance(v, (int, np.integer)) for v in value)
        )

    if B == 1:
        grids_per_batch = [list(image_grids)]
    elif len(image_grids) == B and all(isinstance(v, list) for v in image_grids):
        grids_per_batch = image_grids
    elif not image_grids:
        grids_per_batch = [[] for _ in range(B)]
    else:
        raise ValueError(
            "For batch_size > 1, image_grids must be a list of per-example grid lists."
        )

    def _compute_segment(row, grids, grid_idx):
        """Compute one unpacked document and return h, w, next grid index."""
        seg_len = len(row)
        h = np.zeros(seg_len, dtype=np.int64)
        w = np.zeros(seg_len, dtype=np.int64)
        img_starts = np.where(row == image_start_id)[0]
        img_ends = np.where(row == image_end_id)[0]
        if len(img_starts) != len(img_ends):
            raise ValueError(
                f"Unbalanced image delimiters: starts={len(img_starts)}, ends={len(img_ends)}"
            )

        is_image = np.zeros(seg_len, dtype=np.bool_)
        intervals = []
        previous_end = -1
        for start, end in zip(img_starts, img_ends):
            start, end = int(start), int(end)
            if start <= previous_end or end <= start:
                raise ValueError("Nested or out-of-order image delimiters are not supported.")
            is_image[start + 1 : end] = True
            intervals.append((start, end))
            previous_end = end

        is_visual = is_image & (
            ((row >= visual_offset) & (row < visual_end))
            | (row == mask_token_id)
        )
        is_text = ~is_visual
        text_counter = np.cumsum(is_text)
        text_indices = np.where(is_text)[0]
        text_positions = text_counter - 1
        h[text_indices] = text_positions[text_indices]
        w[text_indices] = text_positions[text_indices]

        for start, end in intervals:
            visual_indices = np.where(is_visual[start + 1 : end])[0] + start + 1
            n_visual = len(visual_indices)
            if n_visual == 0:
                raise ValueError("Image region contains no visual or mask tokens.")

            if grids and grid_idx < len(grids):
                grid = grids[grid_idx]
                if not _is_grid(grid):
                    raise ValueError(f"Invalid image grid: {grid!r}")
                height, width = map(int, grid)
            elif grids:
                raise ValueError(
                    f"Missing image grid for image region {grid_idx}; "
                    f"received only {len(grids)} grids."
                )
            else:
                side = int(np.sqrt(n_visual))
                if side * side != n_visual:
                    raise ValueError(
                        f"Missing image grid for {n_visual} visual tokens; "
                        "only square grids can be inferred."
                    )
                height = width = side
            grid_idx += 1

            if height <= 0 or width <= 0 or height * width != n_visual:
                raise ValueError(
                    f"Image grid {height}x{width} does not match {n_visual} visual tokens."
                )

            base = int(text_counter[start])
            ordinals = np.arange(n_visual)
            h[visual_indices] = base + ordinals // width
            w[visual_indices] = base + ordinals % width

            expected_end_position = base + height
            actual_end_position = int(text_counter[end]) - 1
            offset = expected_end_position - actual_end_position
            if offset:
                text_counter[end:] += offset
                text_positions = text_counter - 1
                remaining_text = text_indices[text_indices >= end]
                h[remaining_text] = text_positions[remaining_text]
                w[remaining_text] = text_positions[remaining_text]

        return h, w, grid_idx

    for b in range(B):
        doc_row = document_np[b]
        if L == 0:
            continue
        boundaries = np.flatnonzero(doc_row[1:] != doc_row[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [L]))
        seen_document_ids = set()
        grid_idx = 0

        for start, end in zip(starts, ends):
            document_id = int(doc_row[start])
            if document_id in seen_document_ids:
                raise ValueError(f"document_id {document_id} is not contiguous.")
            seen_document_ids.add(document_id)
            h, w, grid_idx = _compute_segment(
                input_np[b, start:end], grids_per_batch[b], grid_idx
            )
            h_pos_np[b, start:end] = h
            w_pos_np[b, start:end] = w

        if grids_per_batch[b] and grid_idx != len(grids_per_batch[b]):
            raise ValueError(
                f"Received {len(grids_per_batch[b])} image grids but found "
                f"{grid_idx} image regions in batch item {b}."
            )

    return torch.stack(
        [torch.from_numpy(h_pos_np), torch.from_numpy(w_pos_np)], dim=0
    )  # (2, B, L)
