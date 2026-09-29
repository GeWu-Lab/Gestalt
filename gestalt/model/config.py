"""
Gestalt-DiMOO 模型配置与常量

该模块是全项目特殊 Token ID、词表参数的唯一权威来源。
训练侧数据处理、模型和 tokenizer 均应从此处导入，禁止在各处重复硬编码魔法数字。
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 词表配置
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiMOOVocabConfig:
    """
    Gestalt-DiMOO 词表参数。

    LLaDA-8B 原始词表大小为 126464，VQVAE codebook 共 16384 个 code。
    tokenizer 中视觉 token 覆盖到 142740（exclusive），模型 embedding/output
    则 padding 到 142848，以满足 128 对齐并容纳 interaction token。

    视觉 token 在扩展词表中的 ID 范围：
        [VISUAL_TOKEN_OFFSET, VISUAL_TOKEN_OFFSET + VQVAE_CODEBOOK_SIZE)
        即 [126356, 142740)

    注意：VISUAL_TOKEN_OFFSET(126356) < BASE_VOCAB_SIZE(126464)，
    视觉 token 与文本 token 存在重叠区间，需依靠上下文（BOI/EOI 标记）
    来区分当前位置是文本还是图像 token。
    """
    BASE_VOCAB_SIZE: int = 126464        # LLaDA-8B 原始词表大小
    VQVAE_CODEBOOK_SIZE: int = 16384     # VQVAE codebook 大小
    EXTENDED_VOCAB_SIZE: int = 142848    # 模型词表大小（padding 到 128 的倍数）
    VISUAL_TOKEN_OFFSET: int = 126356    # 视觉 token 在词表中的起始 ID
    VISUAL_TOKEN_END: int = 142740       # 126356 + 16384 = 142740


# ---------------------------------------------------------------------------
# 特殊 Token ID（与 tokenizer.json 对齐）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpecialTokenIds:
    """
    所有特殊 token 的 ID 定义。

    该类是模型、训练和推理共同使用的唯一特殊 token 权威来源。

    Token 说明：
        BOS         : <|startoftext|> — 序列开始
        EOS         : <|endoftext|>   — 序列结束
        NEWLINE     : 换行符，用于保持图像 2D 行列结构
        IMAGE_START : <IMAGE>  — 图像区域起始标记
        IMAGE_END   : </IMAGE> — 图像区域结束标记
        MASK        : <|mdm_mask|> — 离散扩散 mask token
        PADDING     : <padding>    — batch padding token（不参与损失计算）
        ANSWER_START: <answer>     — Stage3 答案区域起始
        ANSWER_END  : </answer>    — Stage3 答案区域结束
    """
    # 文本边界
    BOS: int = 126080           # <|startoftext|>
    EOS: int = 126081           # <|endoftext|>
    NEWLINE: int = 126084       # 换行 token（reserved_token_0）

    # 图像区域标记
    IMAGE_START: int = 126349   # <IMAGE>
    IMAGE_END: int = 126350     # </IMAGE>

    # 扩散 & 填充
    MASK: int = 126336          # <|mdm_mask|>
    PADDING: int = 126339       # <padding>

    # Stage3 答案标记
    ANSWER_START: int = 126354  # <answer>
    ANSWER_END: int = 126355    # </answer>

    # Stage3 系统/用户标记
    SYSTEM_START: int = 126332  # <system>
    SYSTEM_END: int = 126333    # </system>
    USER_START: int = 126334    # <user>
    USER_END: int = 126335      # </user>

    INTERACTION_TOKEN_START: int = 142750  # Interaction token start ID
    INTERACTION_TOKEN_END: int = 142782    # 142750 + 32 = 142782 (32 interaction tokens)
    VISUAL_TOKEN_OFFSET: int = 126356    # 视觉 token 在词表中的起始 ID
    VISUAL_TOKEN_END: int = 142740       # 126356 + 16384 = 142740



# ---------------------------------------------------------------------------
# 模块级单例（方便直接 import 使用）
# ---------------------------------------------------------------------------

# 全局单例，避免到处实例化
VOCAB_CONFIG = DiMOOVocabConfig()
SPECIAL_TOKENS = SpecialTokenIds()


# ---------------------------------------------------------------------------
# 注意力 & 位置编码配置
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttentionConfig:
    use_flex_attention: bool = True    # Use FlexAttention (training) vs SDPA
    use_mrope: bool = True             # Use 2D MRoPE vs 1D RoPE

ATTENTION_CONFIG = AttentionConfig()
