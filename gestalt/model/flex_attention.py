"""FlexAttention mask_mod functions and dense attention mask builder.

Provides parameterized mask_mod functions for CUDA FlexAttention training and
an equivalent dense mask builder for SDPA inference/CPU training. The two
paths produce identical masking behavior.

Mask modes:
- document_mask_mod: bidirectional attention within documents, padding blocked.
- isolated_mask_mod: same as document but also blocks cross-modality attention.
"""

import torch

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    _compiled_flex_attn = torch.compile(flex_attention)
except ImportError:
    flex_attention = None
    create_block_mask = None
    _compiled_flex_attn = None


def _make_document_mask_mod(document_ids, is_padding):
    """Create a document-level bidirectional mask_mod. Blocks cross-document and padding."""
    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_ids[b, q_idx] == document_ids[b, kv_idx]
        not_padding = ~is_padding[b, q_idx] & ~is_padding[b, kv_idx]
        return same_doc & not_padding
    return mask_mod


def _make_isolated_mask_mod(document_ids, is_padding, is_image_token, is_interaction_token):
    """Create a mask_mod that also blocks cross-modality attention."""
    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_ids[b, q_idx] == document_ids[b, kv_idx]
        not_padding = ~is_padding[b, q_idx] & ~is_padding[b, kv_idx]
        q_is_img = is_image_token[b, q_idx]
        k_is_img = is_image_token[b, kv_idx]
        q_is_inter = is_interaction_token[b, q_idx]
        k_is_inter = is_interaction_token[b, kv_idx]
        
        can_attend = (q_is_inter | k_is_inter) | (q_is_img == k_is_img)
        return same_doc & can_attend & not_padding
    return mask_mod


def build_dense_attn_mask(input_ids, document_ids, is_image_token, is_padding, is_interaction_token, isolated=False):
    """Build a dense attention mask equivalent to the FlexAttention mask_mods.

    Args:
        input_ids: (B, L) token ids (used only for shape/device).
        document_ids: (B, L) int tensor, document index per position.
        is_image_token: (B, L) bool tensor, True for image tokens.
        is_padding: (B, L) bool tensor, True for padding positions.
        isolated: if True, also block cross-modality attention.

    Returns:
        (B, 1, L, L) float tensor with 0.0 (attend) / -1e4 (block).
    """
    B, L = input_ids.shape
    mask = torch.zeros(B, 1, L, L, dtype=torch.float, device=input_ids.device)

    doc_q = document_ids.unsqueeze(-1)  # (B, L, 1)
    doc_k = document_ids.unsqueeze(-2)  # (B, 1, L)
    same_doc = doc_q == doc_k  # (B, L, L)

    pad_q = is_padding.unsqueeze(-1)
    pad_k = is_padding.unsqueeze(-2)
    not_padding = ~pad_q & ~pad_k

    allowed = same_doc & not_padding  # (B, L, L)

    if isolated:
        img_q = is_image_token.unsqueeze(-1)
        img_k = is_image_token.unsqueeze(-2)
        
        inter_q = is_interaction_token.unsqueeze(-1) # (B, L, 1)
        inter_k = is_interaction_token.unsqueeze(-2) # (B, 1, L)

        can_attend = (img_q == img_k) | inter_q | inter_k
        
        allowed = allowed & can_attend

    mask = mask.masked_fill(~allowed.unsqueeze(1), -1e4)
    return mask


def create_block_masks(input_ids, document_ids, is_image_token, is_padding, is_interaction_token, n_heads, device):
    """Create FlexAttention BlockMask objects for both masking modes.

    Returns:
        (block_mask_full, block_mask_isolated) tuple.
    """
    if create_block_mask is None:
        raise RuntimeError(
            "FlexAttention is not available. Requires PyTorch with "
            "torch.nn.attention.flex_attention support."
        )

    B, L = input_ids.shape
    doc_mask_mod = _make_document_mask_mod(document_ids, is_padding)
    iso_mask_mod = _make_isolated_mask_mod(document_ids, is_padding, is_image_token, is_interaction_token)

    block_mask_full = create_block_mask(
        doc_mask_mod, B, n_heads, L, L, device=device,
    )
    block_mask_isolated = create_block_mask(
        iso_mask_mod, B, n_heads, L, L, device=device,
    )
    return block_mask_full, block_mask_isolated
