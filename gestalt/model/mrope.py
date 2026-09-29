"""Two-dimensional rotary positions for text-and-image token sequences."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import SPECIAL_TOKENS, VOCAB_CONFIG


class MRoPE2DRotaryEmbedding(nn.Module):
    """Apply 2D RoPE while preserving pretrained 1D text behavior."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rope_theta = config.rope_theta

        dim = config.d_model // config.n_heads
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

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
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float)

        inv_freq_h = inv_freq[0::2]
        inv_freq_w = inv_freq[1::2]

        h_pos = position_ids[0].float()
        w_pos = position_ids[1].float()

        freqs_h = h_pos.unsqueeze(-1) * inv_freq_h
        freqs_w = w_pos.unsqueeze(-1) * inv_freq_w

        # Interleaving is required for exact compatibility when h_pos == w_pos.
        interleaved = torch.stack([freqs_h, freqs_w], dim=-1).flatten(-2)
        positions = torch.cat([interleaved, interleaved], dim=-1)

        pos_sin = positions.sin().unsqueeze(1)
        pos_cos = positions.cos().unsqueeze(1)

        return pos_sin, pos_cos

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply positions shaped (2, B, T) to query and key tensors."""
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
    """Return (height, width) positions shaped (2, B, L).

    Text uses identical height/width positions, image regions use their grids,
    and positions restart at packed-document boundaries.
    """
    if image_grids is None:
        image_grids = []
    B, L = input_ids.shape

    if document_ids is not None and document_ids.shape != input_ids.shape:
        raise ValueError(
            f"document_ids shape {tuple(document_ids.shape)} does not match "
            f"input_ids shape {tuple(input_ids.shape)}"
        )

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
