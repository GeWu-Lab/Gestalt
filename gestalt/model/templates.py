"""Canonical Stage-III prompt templates shared by training and inference."""

from typing import Iterable, List, Sequence

from .config import SPECIAL_TOKENS


DEFAULT_MMU_SYSTEM_PROMPT = (
    "You are a multimodal expert model, which is capable of fully utilizing "
    "multimodal information from both text and images and achieving efficient "
    "interaction between them to accomplish tasks."
)


def interaction_token_ids() -> List[int]:
    return list(range(
        SPECIAL_TOKENS.INTERACTION_TOKEN_START,
        SPECIAL_TOKENS.INTERACTION_TOKEN_END,
    ))


def text_span(token_ids: Sequence[int]) -> List[int]:
    if not token_ids:
        return []
    return [SPECIAL_TOKENS.BOS, *map(int, token_ids), SPECIAL_TOKENS.EOS]


def build_mmu_condition(
    system_tokens: Sequence[int],
    image_token_groups: Iterable[Sequence[int]],
    question_tokens: Sequence[int],
) -> List[int]:
    """Build the condition preceding ``<answer>`` for I2T/MMU."""
    image_tokens: List[int] = []
    for group in image_token_groups:
        group = list(map(int, group))
        if not group:
            raise ValueError("MMU image token groups must not be empty.")
        image_tokens.extend([
            SPECIAL_TOKENS.IMAGE_START,
            *group,
            SPECIAL_TOKENS.IMAGE_END,
        ])
    if not image_tokens:
        raise ValueError("MMU requires at least one image token group.")

    return [
        SPECIAL_TOKENS.SYSTEM_START,
        *map(int, system_tokens),
        SPECIAL_TOKENS.SYSTEM_END,
        SPECIAL_TOKENS.USER_START,
        *image_tokens,
        *interaction_token_ids(),
        *text_span(question_tokens),
        SPECIAL_TOKENS.USER_END,
    ]


def build_t2i_condition(
    system_tokens: Sequence[int],
    user_tokens: Sequence[int],
) -> List[int]:
    """Build the condition preceding ``<answer><IMAGE>`` for T2I."""
    return [
        SPECIAL_TOKENS.SYSTEM_START,
        *map(int, system_tokens),
        SPECIAL_TOKENS.SYSTEM_END,
        SPECIAL_TOKENS.USER_START,
        *text_span(user_tokens),
        SPECIAL_TOKENS.USER_END,
        *interaction_token_ids(),
    ]


def build_i2i_condition(
    system_tokens: Sequence[int],
    input_image_tokens: Sequence[int],
    user_tokens: Sequence[int],
) -> List[int]:
    """Build the condition preceding ``<answer><IMAGE>`` for I2I."""
    if not input_image_tokens:
        raise ValueError("I2I requires input image tokens.")
    return [
        SPECIAL_TOKENS.SYSTEM_START,
        *map(int, system_tokens),
        SPECIAL_TOKENS.SYSTEM_END,
        SPECIAL_TOKENS.USER_START,
        SPECIAL_TOKENS.IMAGE_START,
        *map(int, input_image_tokens),
        SPECIAL_TOKENS.IMAGE_END,
        *text_span(user_tokens),
        SPECIAL_TOKENS.USER_END,
        *interaction_token_ids(),
    ]
