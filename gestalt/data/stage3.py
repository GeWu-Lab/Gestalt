"""
Stage-III Supervised Fine-Tuning: Instruction Following

Multi-round conversation processor with interaction tokens.

Supports three task types:
1. I2T (Image-to-Text): Understanding tasks with multi-round conversations
   Input format from parquet (new multiround format):
       {
           "conversations": JSON string: [{"from": "human"/"gpt", "value": str}, ...],
           "img_tokens": np.ndarray (raw codebook IDs, not offset),
           "metadata": {"token_height": int, "token_width": int, ...}
       }
   Sequence format (single image):
       <system> {system_prompt} </system>
       <user><IMAGE> {img_tokens} </IMAGE>[INTERACTION*32] BOS {human_1} EOS </user>
       <answer> BOS {gpt_1} EOS </answer>
       ...
   Sequence format (multiple images):
       <system> {system_prompt} </system>
       <user><IMAGE> {img1_tokens} </IMAGE><IMAGE> {img2_tokens} </IMAGE>[INTERACTION*32] BOS {human_1} EOS </user>
       <answer> BOS {gpt_1} EOS </answer>
       ...
   Loss: Only gpt answer text tokens and answer-side EOS are masked for loss computation

2. T2I (Text-to-Image): Generation tasks with text-to-image
   Input format from parquet:
       {
           "system_prompt": str,
           "user_prompt": str,
           "answer_image": {"img_tokens": np.ndarray},
           "metadata": dict
       }
   Sequence format:
       <system> {system_prompt} </system> <user> BOS {user_prompt} EOS </user> [INTERACTION*32]<answer> <IMAGE> {img_tokens} </IMAGE></answer>
   Loss: Only image tokens are masked for loss computation

3. I2I (Image-to-Image): Editing tasks with input image and output image
   Input format from parquet:
       {
           "system_prompt": str,
           "user_prompt": str,
           "input_image": {"img_tokens": np.ndarray},
           "answer_image": {"img_tokens": np.ndarray},
           "metadata": dict
       }
   Sequence format:
       <system> {system_prompt} </system> <user> <IMAGE> {input_img_tokens} </IMAGE>BOS {user_prompt} EOS </user> [INTERACTION*32] <answer> <IMAGE> {output_img_tokens} </IMAGE> </answer>
   Loss: Only output image tokens are masked for loss computation

Output format consumed by the Stage-III collator:
    {
        "inputs_id": original_input_ids,        # 原始 ID
        "masked_inputs_id": masked_input_ids,   # 喂给模型的 ID
        "loss_mask": loss_mask,                 # 哪些位置需要计算 Loss (bool)
        "labels": labels,                        # 传统的 labels 格式 (-100 for non-loss)
        "ce_loss_weights": ce_loss_weights,      # 预训练式 token-level loss 权重
    }
"""

import json
import math
import numbers
import random
import re
from typing import Dict, List, Tuple, Optional, Set, Union

from gestalt.model.config import SpecialTokenIds, VOCAB_CONFIG
from gestalt.model.templates import DEFAULT_MMU_SYSTEM_PROMPT, interaction_token_ids



class Stage3Processor:
    """
    Stage-III Data Processor for multi-round conversations and text-to-image tasks.

    Features:
    - Handles I2T: multi-round human/gpt conversations with loss on answer_text
    - Handles T2I: text-to-image generation with loss on image tokens
    - Inserts 32 interaction tokens between text and image
    - Returns the masked-token format consumed by Stage-III training
    """

    INTERACTION_TOKENS = interaction_token_ids()

    # Default system prompt for I2T
    SYSTEM_PROMPT_I2T = DEFAULT_MMU_SYSTEM_PROMPT

    HUMAN_ROLES: Set[str] = {"human", "user"}
    GPT_ROLES: Set[str] = {"gpt", "assistant", "model"}
    def __init__(
        self,
        tokenizer,
        max_length: int = 5120,
        answer_mask_schedule: str = "linear",
        image_mask_schedule: str = "linear",
        image_loss_weight: float = 1.0,
        loss_weight_mode: str = "sample",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_mask_schedule = answer_mask_schedule
        self.image_mask_schedule = image_mask_schedule
        self.image_loss_weight = image_loss_weight
        self.loss_weight_mode = loss_weight_mode
        if max_length <= 0:
            raise ValueError("max_length must be positive.")
        valid_schedules = {"cosine", "linear", "uniform"}
        if answer_mask_schedule not in valid_schedules:
            raise ValueError(f"Unknown answer_mask_schedule={answer_mask_schedule!r}.")
        if image_mask_schedule not in valid_schedules:
            raise ValueError(f"Unknown image_mask_schedule={image_mask_schedule!r}.")
        if image_loss_weight <= 0:
            raise ValueError("image_loss_weight must be positive.")
        if loss_weight_mode not in {"sample", "token"}:
            raise ValueError("loss_weight_mode must be 'sample' or 'token'.")
        # 独立 RNG，避免使用全局 random（跨 rank 种子相同）
        self.rng = random.Random()

        # Cache for system prompts
        self._system_prompt_tokens_i2t = None
        self._system_prompt_tokens_cache = {}  # Cache for custom system prompts

    @staticmethod
    def _ce_weights_from_labels(
        labels: List[int],
        segment_len: int,
        mask_ratio: float,
        multiplier: float = 1.0,
        mode: str = "sample",
    ) -> List[float]:
        if not labels:
            return []
        if mode == "token":
            weight = (1.0 / math.sqrt(max(segment_len, 1))) / max(mask_ratio, 1e-6)
            weight *= multiplier
        else:
            num_masked = sum(1 for lbl in labels if lbl != -100)
            if num_masked == 0:
                return [0.0] * len(labels)
            weight = multiplier / num_masked
        return [weight if label != -100 else 0.0 for label in labels]

    def _extend_text_span(
        self,
        tokens: List[int],
        extend_fn,
        *,
        is_loss: bool = False,
        multiplier: float = 1.0,
    ) -> Optional[float]:
        """Append a BOS/EOS-wrapped text span.

        BOS is never a prediction target. For loss spans, the content tokens and
        the trailing EOS are denoising targets under the normal mask schedule.
        For non-loss spans, BOS/content/EOS are all clean conditioning tokens.

        Returns the sampled mask_ratio when ``is_loss=True``, else ``None``.
        """
        if not tokens:
            return None

        extend_fn(
            [SpecialTokenIds.BOS],
            [SpecialTokenIds.BOS],
            [False],
        )

        bounded = list(tokens) + [SpecialTokenIds.EOS]
        if is_loss:
            masked, labels, mask_ratio = self._mask_answer_text(bounded)
            weights = self._ce_weights_from_labels(
                labels,
                segment_len=len(bounded),
                mask_ratio=mask_ratio,
                multiplier=multiplier,
                mode=self.loss_weight_mode,
            )
            extend_fn(
                bounded,
                masked,
                [lbl != -100 for lbl in labels],
                weights=weights,
            )
            return mask_ratio
        else:
            extend_fn(
                bounded,
                bounded,
                [False] * len(bounded),
            )
            return None

    @staticmethod
    def _truncation_detail(task: str, original_len: int, max_length: int) -> str:
        return f"task={task} | seq_len={original_len} | max_length={max_length}"

    @staticmethod
    def _record_processor_event(raw_data: Dict, event: str, detail: str):
        raw_data["__stage3_processor_event__"] = {
            "event": event,
            "detail": detail,
        }

    def seed_rng(self, seed: int):
        """用 global_worker_id 重新种子化，确保不同 rank/worker 的 mask ratio 独立。"""
        self.rng = random.Random(seed)

    @staticmethod
    def _strip_image_placeholders(text: str) -> str:
        """Remove textual image placeholders after image tokens are inserted explicitly."""
        if not text:
            return text
        text = re.sub(r"(?i)<\s*/?\s*image\s*>", "", text)
        text = re.sub(r"(?i)<\s*img_context\s*>", "", text)
        return text.lstrip()

    @staticmethod
    def _normalize_role(turn: Dict) -> Optional[str]:
        raw_role = str(turn.get("from", turn.get("role", ""))).lower().strip()
        if raw_role in Stage3Processor.HUMAN_ROLES:
            return "human"
        if raw_role in Stage3Processor.GPT_ROLES:
            return "gpt"
        return None

    @staticmethod
    def _turn_text(turn: Dict) -> str:
        return turn.get("value", turn.get("content", "")) or ""

    @staticmethod
    def _maybe_complete_split_question(text: str) -> str:
        """Fill in terse region-only turns after splitting multiturn QA.

        Some real multiturn rows use the first turn to state the region-caption
        instruction, then later user turns contain only a bbox like
        "[0.026,0.195,0.998,0.357]".  After split_qa those turns need a
        self-contained prompt.
        """
        stripped = text.strip()
        bbox_only = re.fullmatch(
            r"\[?\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*"
            r"-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]?",
            stripped,
        )
        if bbox_only:
            return f"Please provide a short description for this region: {stripped}."
        return text

    def _decode_conversations(self, conversations) -> List[Dict]:
        if isinstance(conversations, str):
            conversations = json.loads(conversations)
        if not isinstance(conversations, list):
            return []
        return conversations

    @staticmethod
    def _normalize_image_token_sequence(tokens, field_name: str) -> List[int]:
        """Return model token IDs and reject mixed or out-of-range codebooks."""
        if not all(isinstance(token, numbers.Integral) and not isinstance(token, bool) for token in tokens):
            raise ValueError(f"{field_name} must contain integer token IDs.")
        values = [int(token) for token in tokens]
        if not values:
            return []

        codebook_size = VOCAB_CONFIG.VQVAE_CODEBOOK_SIZE
        visual_start = VOCAB_CONFIG.VISUAL_TOKEN_OFFSET
        visual_end = VOCAB_CONFIG.VISUAL_TOKEN_END
        if all(0 <= token < codebook_size for token in values):
            return [token + visual_start for token in values]
        if all(visual_start <= token < visual_end for token in values):
            return values
        raise ValueError(
            f"{field_name} must contain either raw VQVAE IDs in [0, {codebook_size}) "
            f"or model visual-token IDs in [{visual_start}, {visual_end})."
        )

    @staticmethod
    def _prepare_image_tokens(img_tokens) -> List[List[int]]:
        """Prepare image tokens, preserving per-image boundaries.

        Returns a list of per-image token lists so each image can be wrapped
        in its own <IMAGE>...</IMAGE> pair.
        """
        if img_tokens is None or len(img_tokens) == 0:
            return []
        img_tokens = list(img_tokens)
        if img_tokens and not isinstance(img_tokens[0], numbers.Number):
            result = []
            for index, per_image in enumerate(img_tokens):
                per_image = Stage3Processor._normalize_image_token_sequence(
                    per_image, f"img_tokens[{index}]"
                )
                if not per_image:
                    continue
                result.append(per_image)
            return result
        return [Stage3Processor._normalize_image_token_sequence(img_tokens, "img_tokens")]

    def _get_system_prompt_tokens(self, system_prompt: Optional[str] = None) -> List[int]:
        """Get cached system prompt tokens."""
        if system_prompt is None:
            system_prompt = self.SYSTEM_PROMPT_I2T

        # Check if default I2T prompt
        if system_prompt == self.SYSTEM_PROMPT_I2T:
            if self._system_prompt_tokens_i2t is None:
                self._system_prompt_tokens_i2t = self.tokenizer(
                    system_prompt,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length,
                )["input_ids"]
            return self._system_prompt_tokens_i2t

        # Cache custom prompts
        if system_prompt not in self._system_prompt_tokens_cache:
            self._system_prompt_tokens_cache[system_prompt] = self.tokenizer(
                system_prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]
        return self._system_prompt_tokens_cache[system_prompt]

    def __call__(self, raw_data: Dict) -> Optional[Union[Dict[str, List], List[Dict[str, List]]]]:
        """
        Process raw parquet data into training format.

        Automatically detects task type based on data structure.
        Unified-format data (with explicit task_type) is converted to the
        expected field layout by Stage3Dataset before reaching this method,
        so routing here is purely field-based.

        Args:
            raw_data: Dict with varying structure based on task type

        Returns:
            Processed dict with inputs_id, masked_inputs_id, loss_mask, labels, ce_loss_weights
            or None if processing fails
        """
        # Detect task type based on data structure.
        has_input_image = "input_image" in raw_data
        has_answer_image = "answer_image" in raw_data
        has_metadata_top = "metadata" in raw_data and isinstance(raw_data.get("metadata"), dict)
        has_conversations = "conversations" in raw_data

        if has_input_image and has_answer_image:
            return self._process_i2i(raw_data)

        if has_answer_image and has_metadata_top:
            return self._process_t2i(raw_data)

        if has_conversations:
            return self._process_i2t_split_qa(raw_data)

        raise ValueError(
            "Cannot infer task type: expected conversations (I2T), answer_image "
            "(T2I), or input_image + answer_image (I2I)."
        )

    def _process_i2t_split_qa(self, raw_data: Dict) -> Optional[List[Dict[str, List]]]:
        """
        Split a multiturn I2T conversation into independent image-grounded QA samples.

        Original:
            <image> Q1 -> A1, Q2 -> A2, ...

        Returned samples:
            sample 1: <image> Q1 -> A1
            sample 2: <image> Q2 -> A2
            ...

        This avoids bidirectional-attention leakage from future turns while
        preserving loss on every assistant answer.
        """
        system_prompt = raw_data.get("system_prompt", self.SYSTEM_PROMPT_I2T)
        conversations = self._decode_conversations(raw_data.get("conversations", []))
        img_token_groups = self._prepare_image_tokens(raw_data.get("img_tokens"))
        if not img_token_groups:
            raise ValueError("I2T sample has no image tokens.")

        samples: List[Dict[str, List]] = []
        pending_question: Optional[str] = None

        for turn in conversations:
            role = self._normalize_role(turn)
            if role is None:
                raise ValueError(f"Unsupported I2T conversation role: {turn!r}")

            text = self._turn_text(turn)

            if role == "human":
                if pending_question is not None:
                    raise ValueError("I2T conversation has consecutive user turns.")
                if img_token_groups:
                    text = self._strip_image_placeholders(text)
                text = self._maybe_complete_split_question(text)
                pending_question = text
                continue

            if role == "gpt":
                answer = text.strip()
                if pending_question is None:
                    raise ValueError("I2T conversation has an answer without a user question.")
                if not answer:
                    raise ValueError("I2T assistant answer is empty.")

                sample = self._build_i2t_single_qa(
                    raw_data=raw_data,
                    system_prompt=system_prompt,
                    question=pending_question,
                    answer=answer,
                    img_token_groups=img_token_groups,
                )
                pending_question = None
                if sample is not None:
                    samples.append(sample)

        if pending_question is not None:
            raise ValueError("I2T conversation ends with an unanswered user question.")
        return samples or None

    def _build_i2t_single_qa(
        self,
        raw_data: Dict,
        system_prompt: str,
        question: str,
        answer: str,
        img_token_groups: List[List[int]],
    ) -> Optional[Dict[str, List]]:
        """Build one independent I2T QA sample with image + interaction tokens."""
        original_tokens = []
        masked_tokens = []
        loss_mask = []
        ce_loss_weights = []

        def _append(tok, is_loss=False, weight=0.0):
            original_tokens.append(tok)
            masked_tokens.append(tok)
            loss_mask.append(is_loss)
            ce_loss_weights.append(weight)

        def _extend(toks, masked_toks, loss_flags, weights=None):
            original_tokens.extend(toks)
            masked_tokens.extend(masked_toks)
            loss_mask.extend(loss_flags)
            ce_loss_weights.extend(weights if weights is not None else [0.0] * len(toks))

        # <system> {system_prompt} </system>
        _append(SpecialTokenIds.SYSTEM_START)
        sys_ids = self._get_system_prompt_tokens(system_prompt)
        _extend(sys_ids, sys_ids, [False] * len(sys_ids))
        _append(SpecialTokenIds.SYSTEM_END)

        # <user> <IMAGE>...</IMAGE> [<IMAGE>...</IMAGE>...] [INTERACTION*32] BOS {question} EOS </user>
        _append(SpecialTokenIds.USER_START)
        if img_token_groups:
            for single_img in img_token_groups:
                _append(SpecialTokenIds.IMAGE_START)
                _extend(single_img, single_img, [False] * len(single_img))
                _append(SpecialTokenIds.IMAGE_END)
            original_tokens.extend(self.INTERACTION_TOKENS)
            masked_tokens.extend(self.INTERACTION_TOKENS)
            loss_mask.extend([False] * 32)
            ce_loss_weights.extend([0.0] * 32)

        if question:
            question_ids = self.tokenizer(
                question,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]
            self._extend_text_span(question_ids, _extend, is_loss=False)
        _append(SpecialTokenIds.USER_END)

        # <answer> BOS {answer} EOS </answer>
        answer_ids = self.tokenizer(
            answer,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        if not answer_ids:
            return None
        _append(SpecialTokenIds.ANSWER_START)
        self._extend_text_span(answer_ids, _extend, is_loss=True)
        _append(SpecialTokenIds.ANSWER_END)

        if len(original_tokens) > self.max_length:
            self._record_processor_event(
                raw_data,
                "DROP_OVER_LENGTH",
                self._truncation_detail(
                    "i2t_split_qa", len(original_tokens), self.max_length
                ),
            )
            return None

        labels = [orig if is_loss else -100 for orig, is_loss in zip(original_tokens, loss_mask)]
        if not any(label != -100 for label in labels):
            return None

        sample = {
            "inputs_id": original_tokens,
            "masked_inputs_id": masked_tokens,
            "loss_mask": loss_mask,
            "labels": labels,
            "ce_loss_weights": ce_loss_weights,
        }
        return sample

    def _process_t2i(self, raw_data: Dict) -> Optional[Dict[str, List]]:
        """
        Process T2I (Text-to-Image) data.

        Input format:
            {
                "system_prompt": str,
                "user_prompt": str,
                "answer_image": {"img_tokens": np.ndarray},
                "metadata": dict
            }

        Sequence format:
            <system> {system_prompt} </system>
            <user> BOS {user_prompt} EOS </user>
            [INTERACTION*32]<answer> <IMAGE> {img_tokens} </IMAGE></answer>

        Only image tokens are masked for loss computation.
        """
        # Extract fields
        system_prompt = raw_data.get("system_prompt", "")
        user_prompt = raw_data.get("user_prompt", "")
        answer_image = raw_data.get("answer_image", {})
        # Extract image tokens
        if isinstance(answer_image, dict):
            img_tokens = answer_image.get("img_tokens")
        else:
            img_tokens = None

        # Process image tokens
        img_tokens = self._normalize_image_token_sequence(
            [] if img_tokens is None else img_tokens, "answer_image.img_tokens"
        )
        if not img_tokens:
            raise ValueError("T2I sample has no answer image tokens.")

        # Mask image tokens
        masked_img_tokens, img_labels, img_mask_ratio = self._mask_image_tokens(img_tokens)
        img_weights = self._ce_weights_from_labels(
            img_labels,
            segment_len=len(img_tokens),
            mask_ratio=img_mask_ratio,
            multiplier=self.image_loss_weight,
            mode=self.loss_weight_mode,
        )

        # Build sequence
        original_tokens = []
        masked_tokens = []
        loss_mask = []
        ce_loss_weights = []

        def _append(tok, is_loss=False, weight=0.0):
            original_tokens.append(tok)
            masked_tokens.append(tok)
            loss_mask.append(is_loss)
            ce_loss_weights.append(weight)

        def _extend(toks, masked_toks, loss_flags, weights=None):
            original_tokens.extend(toks)
            masked_tokens.extend(masked_toks)
            loss_mask.extend(loss_flags)
            ce_loss_weights.extend(weights if weights is not None else [0.0] * len(toks))

        # Add <system> {system_prompt} </system>
        _append(SpecialTokenIds.SYSTEM_START)

        system_prompt_ids = self._get_system_prompt_tokens(system_prompt)
        _extend(system_prompt_ids, system_prompt_ids, [False] * len(system_prompt_ids))
        _append(SpecialTokenIds.SYSTEM_END)

        # Add <user> BOS {user_prompt} EOS </user>
        _append(SpecialTokenIds.USER_START)

        user_prompt_ids = self.tokenizer(
            user_prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        self._extend_text_span(user_prompt_ids, _extend, is_loss=False)
        _append(SpecialTokenIds.USER_END)

        # Add interaction tokens (32 tokens)
        original_tokens.extend(self.INTERACTION_TOKENS)
        masked_tokens.extend(self.INTERACTION_TOKENS)
        loss_mask.extend([False] * 32)
        ce_loss_weights.extend([0.0] * 32)

        # Add <answer> <IMAGE> {img_tokens} </IMAGE></answer>
        _append(SpecialTokenIds.ANSWER_START)

        if img_tokens:
            # IMAGE_START
            _append(SpecialTokenIds.IMAGE_START)

            # Image tokens (masked for T2I)
            # Convert labels from token_id/-100 to bool (True if loss position)
            img_loss_mask = [lbl != -100 for lbl in img_labels]
            original_tokens.extend(img_tokens)
            masked_tokens.extend(masked_img_tokens)
            loss_mask.extend(img_loss_mask)
            ce_loss_weights.extend(img_weights)

            # IMAGE_END
            _append(SpecialTokenIds.IMAGE_END)

        _append(SpecialTokenIds.ANSWER_END)

        if len(original_tokens) > self.max_length:
            self._record_processor_event(
                raw_data,
                "DROP_OVER_LENGTH",
                self._truncation_detail("t2i", len(original_tokens), self.max_length),
            )
            return None

        # Build labels from loss_mask
        labels = []
        for orig, is_loss in zip(original_tokens, loss_mask):
            labels.append(orig if is_loss else -100)

        sample = {
            "inputs_id": original_tokens,
            "masked_inputs_id": masked_tokens,
            "loss_mask": loss_mask,
            "labels": labels,
            "ce_loss_weights": ce_loss_weights,
        }
        return sample

    def _process_i2i(self, raw_data: Dict) -> Optional[Dict[str, List]]:
        """
        Process I2I (Image-to-Image) editing data.

        Input format:
            {
                "system_prompt": str,
                "user_prompt": str,
                "input_image": {"img_tokens": np.ndarray},
                "answer_image": {"img_tokens": np.ndarray},
                "metadata": dict
            }

        Sequence format:
            <system> {system_prompt} </system>
            <user> <IMAGE> {input_img_tokens} </IMAGE>BOS {user_prompt} EOS </user>
            [INTERACTION*32]
            <answer> <IMAGE> {output_img_tokens} </IMAGE> </answer>

        Loss: Only output image tokens (answer_image) are masked for loss computation.
        """
        # Extract fields
        system_prompt = raw_data.get("system_prompt", "")
        user_prompt = raw_data.get("user_prompt", "")
        input_image = raw_data.get("input_image", {})
        answer_image = raw_data.get("answer_image", {})
        # Extract and offset input image tokens (condition, not masked)
        if isinstance(input_image, dict):
            in_tokens = input_image.get("img_tokens")
        else:
            in_tokens = None

        in_tokens = self._normalize_image_token_sequence(
            [] if in_tokens is None else in_tokens, "input_image.img_tokens"
        )

        # Extract and offset output image tokens (target, masked for loss)
        if isinstance(answer_image, dict):
            out_tokens = answer_image.get("img_tokens")
        else:
            out_tokens = None

        out_tokens = self._normalize_image_token_sequence(
            [] if out_tokens is None else out_tokens, "answer_image.img_tokens"
        )
        if not in_tokens or not out_tokens:
            raise ValueError("I2I sample requires both input and answer image tokens.")

        # Mask output image tokens for loss
        masked_out_tokens, out_labels, out_mask_ratio = self._mask_image_tokens(out_tokens)
        out_weights = self._ce_weights_from_labels(
            out_labels,
            segment_len=len(out_tokens),
            mask_ratio=out_mask_ratio,
            multiplier=self.image_loss_weight,
            mode=self.loss_weight_mode,
        )

        # Build sequence
        original_tokens = []
        masked_tokens = []
        loss_mask = []
        ce_loss_weights = []

        def _append(tok, is_loss=False, weight=0.0):
            original_tokens.append(tok)
            masked_tokens.append(tok)
            loss_mask.append(is_loss)
            ce_loss_weights.append(weight)

        def _extend(toks, masked_toks, loss_flags, weights=None):
            original_tokens.extend(toks)
            masked_tokens.extend(masked_toks)
            loss_mask.extend(loss_flags)
            ce_loss_weights.extend(weights if weights is not None else [0.0] * len(toks))

        # <system> {system_prompt} </system>
        _append(SpecialTokenIds.SYSTEM_START)
        sys_ids = self._get_system_prompt_tokens(system_prompt)
        _extend(sys_ids, sys_ids, [False] * len(sys_ids))
        _append(SpecialTokenIds.SYSTEM_END)

        # <user> <IMAGE> {input_img_tokens} </IMAGE> BOS {user_prompt} EOS </user>
        _append(SpecialTokenIds.USER_START)

        if in_tokens:
            _append(SpecialTokenIds.IMAGE_START)
            _extend(in_tokens, in_tokens, [False] * len(in_tokens))
            _append(SpecialTokenIds.IMAGE_END)

        user_ids = self.tokenizer(
            user_prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        self._extend_text_span(user_ids, _extend, is_loss=False)

        _append(SpecialTokenIds.USER_END)

        # [INTERACTION * 32]
        original_tokens.extend(self.INTERACTION_TOKENS)
        masked_tokens.extend(self.INTERACTION_TOKENS)
        loss_mask.extend([False] * 32)
        ce_loss_weights.extend([0.0] * 32)

        # <answer> <IMAGE> {output_img_tokens} </IMAGE> </answer>
        _append(SpecialTokenIds.ANSWER_START)

        if out_tokens:
            _append(SpecialTokenIds.IMAGE_START)
            out_loss_flags = [lbl != -100 for lbl in out_labels]
            _extend(out_tokens, masked_out_tokens, out_loss_flags, weights=out_weights)
            _append(SpecialTokenIds.IMAGE_END)

        _append(SpecialTokenIds.ANSWER_END)

        if len(original_tokens) > self.max_length:
            self._record_processor_event(
                raw_data,
                "DROP_OVER_LENGTH",
                self._truncation_detail("i2i", len(original_tokens), self.max_length),
            )
            return None

        # Build labels from loss_mask
        labels = [orig if is_loss else -100 for orig, is_loss in zip(original_tokens, loss_mask)]

        sample = {
            "inputs_id": original_tokens,
            "masked_inputs_id": masked_tokens,
            "loss_mask": loss_mask,
            "labels": labels,
            "ce_loss_weights": ce_loss_weights,
        }
        return sample

    def _sample_mask_ratio(self, schedule: str, n: int) -> float:
        """
        Sample a mask ratio according to the given schedule.

        Supported masking schedules:
            cosine  : mask_ratio = cos(r * π/2),  r ~ U(0,1)  → bias toward high ratios
            linear  : mask_ratio = max(r, 0.05),  r ~ U(0,1)  → uniform with 5% floor
            uniform : mask_ratio ~ U(0.3, 0.9)

        Very short sequences (n <= 10) are always fully masked regardless of schedule.

        Args:
            schedule: One of "cosine", "linear", "uniform"
            n: Number of tokens (used for short-sequence guard)

        Returns:
            mask_ratio in (0, 1]
        """
        if n <= 12:
            return 1.0

        r = self.rng.uniform(0, 1)
        if schedule == "cosine":
            return math.cos(r * math.pi / 2)
        elif schedule == "linear":
            return max(r, 0.1)
        elif schedule == "uniform":
            return self.rng.uniform(0.3, 0.9)
        else:
            raise ValueError(f"Unknown mask schedule: {schedule!r}. Choose from 'cosine', 'linear', 'uniform'.")

    def _mask_image_tokens(self, tokens: List[int]) -> Tuple[List[int], List[int], float]:
        """
        Mask image tokens for T2I / I2I tasks.

        Args:
            tokens: List of image token IDs

        Returns:
            Tuple of (masked_tokens, labels, mask_ratio)
        """
        if not tokens:
            return [], [], 0.0

        mask_ratio = self._sample_mask_ratio(self.image_mask_schedule, len(tokens))
        num_to_mask = max(int(len(tokens) * mask_ratio), 1)
        indices = set(self.rng.sample(range(len(tokens)), num_to_mask))

        masked = tokens[:]
        labels = [-100] * len(tokens)
        for idx in indices:
            labels[idx] = tokens[idx]
            masked[idx] = SpecialTokenIds.MASK

        return masked, labels, mask_ratio

    def _mask_answer_text(self, tokens: List[int]) -> Tuple[List[int], List[int], float]:
        """
        Mask answer text tokens for I2T tasks.

        Args:
            tokens: List of text token IDs

        Returns:
            Tuple of (masked_tokens, labels, mask_ratio)
        """
        if not tokens:
            return [], [], 0.0

        mask_ratio = self._sample_mask_ratio(self.answer_mask_schedule, len(tokens))
        num_to_mask = max(int(len(tokens) * mask_ratio), 1)
        indices = set(self.rng.sample(range(len(tokens)), num_to_mask))

        masked = tokens[:]
        labels = [-100] * len(tokens)
        for idx in indices:
            labels[idx] = tokens[idx]
            masked[idx] = SpecialTokenIds.MASK

        return masked, labels, mask_ratio

def create_stage3_processor(
    tokenizer,
    max_length: int = 5120,
    answer_mask_schedule: str = "linear",
    image_mask_schedule: str = "linear",
    image_loss_weight: float = 1.0,
    loss_weight_mode: str = "sample",
) -> Stage3Processor:
    return Stage3Processor(
        tokenizer=tokenizer,
        max_length=max_length,
        answer_mask_schedule=answer_mask_schedule,
        image_mask_schedule=image_mask_schedule,
        image_loss_weight=image_loss_weight,
        loss_weight_mode=loss_weight_mode,
    )
