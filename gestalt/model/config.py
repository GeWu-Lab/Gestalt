"""Canonical vocabulary and special-token constants for Gestalt-DiMOO."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DiMOOVocabConfig:
    """Vocabulary layout shared by data processing, training, and inference.

    Visual IDs overlap the tail of the base text vocabulary, so IMAGE_START
    and IMAGE_END delimiters determine whether an overlapping ID is visual.
    The model vocabulary is padded to 142848 to include interaction-token IDs.
    """
    BASE_VOCAB_SIZE: int = 126464
    VQVAE_CODEBOOK_SIZE: int = 16384
    EXTENDED_VOCAB_SIZE: int = 142848
    VISUAL_TOKEN_OFFSET: int = 126356
    VISUAL_TOKEN_END: int = 142740


@dataclass(frozen=True)
class SpecialTokenIds:
    """Special-token IDs that must match tokenizer.json."""
    BOS: int = 126080
    EOS: int = 126081
    NEWLINE: int = 126084
    IMAGE_START: int = 126349
    IMAGE_END: int = 126350
    MASK: int = 126336
    PADDING: int = 126339
    ANSWER_START: int = 126354
    ANSWER_END: int = 126355
    SYSTEM_START: int = 126332
    SYSTEM_END: int = 126333
    USER_START: int = 126334
    USER_END: int = 126335
    INTERACTION_TOKEN_START: int = 142750
    INTERACTION_TOKEN_END: int = 142782
    VISUAL_TOKEN_OFFSET: int = 126356
    VISUAL_TOKEN_END: int = 142740


VOCAB_CONFIG = DiMOOVocabConfig()
SPECIAL_TOKENS = SpecialTokenIds()


@dataclass(frozen=True)
class AttentionConfig:
    use_flex_attention: bool = True
    use_mrope: bool = True

ATTENTION_CONFIG = AttentionConfig()
