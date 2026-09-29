from .configuration_llada import LLaDAConfig
import torch
import torch.nn as nn
import logging
from typing import Optional, List
from .modeling_llada import LLaDAModelLM
logger = logging.getLogger(__name__)
import math
from gestalt.model.config import SPECIAL_TOKENS, VOCAB_CONFIG
from gestalt.model.mrope import compute_2d_position_ids
from gestalt.model.templates import build_mmu_condition, build_t2i_condition
import torch.nn.functional as F

MASK = SPECIAL_TOKENS.MASK
BOI = SPECIAL_TOKENS.IMAGE_START
EOI = SPECIAL_TOKENS.IMAGE_END
BOS = SPECIAL_TOKENS.BOS
EOS = SPECIAL_TOKENS.EOS
BOA = SPECIAL_TOKENS.ANSWER_START
EOA = SPECIAL_TOKENS.ANSWER_END
INTERACTION_START = SPECIAL_TOKENS.INTERACTION_TOKEN_START
INTERACTION_END = SPECIAL_TOKENS.INTERACTION_TOKEN_END

class GestaltConfig(LLaDAConfig):
    model_type = "gestalt"

    def __init__(
        self,
        extended_vocab_size=VOCAB_CONFIG.EXTENDED_VOCAB_SIZE,
        inter_layer_num=8,
        inter_layer_state="interaction_token",
        **kwargs
    ):
        use_interaction_embedding_adapter = kwargs.get(
            "use_interaction_embedding_adapter",
            False,
        )
        kwargs["tie_word_embeddings"] = kwargs.get("tie_word_embeddings", False)
        super().__init__(**kwargs)
        
        self.extended_vocab_size = extended_vocab_size
        self.inter_tokens_num = (
            SPECIAL_TOKENS.INTERACTION_TOKEN_END
            - SPECIAL_TOKENS.INTERACTION_TOKEN_START
        )
        self.inter_layer_num = inter_layer_num
        self.inter_layer_state = inter_layer_state
        self.use_interaction_embedding_adapter = use_interaction_embedding_adapter


class GestaltModelLM(LLaDAModelLM):
    config_class = GestaltConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: GestaltConfig):
        super().__init__(config)
        logger.debug("Gestalt interaction tokens: %s, d_model: %s", config.inter_tokens_num, config.d_model)

        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        if getattr(config, "use_interaction_embedding_adapter", False):
            self.interaction_embedding_adapter = nn.Embedding(
                config.inter_tokens_num,
                config.d_model,
            )

    def tie_weights(self, missing_keys=None, **kwargs):
        pass
            
    def resize_and_initialize_vocab(
        self,
        target_vocab_size: int,
        init_strategy: str = "mean"
    ):
        wte = self.get_input_embeddings()
        ff_out = self.get_output_embeddings()
        old_input_size = wte.weight.shape[0]
        old_output_size = ff_out.weight.shape[0] if ff_out is not None else old_input_size
        current_vocab_size = old_input_size
        if old_output_size != old_input_size:
            raise ValueError(
                f"Input/output vocab sizes differ: {old_input_size} vs {old_output_size}."
            )
        if self.config.vocab_size != old_input_size:
            raise ValueError(
                f"Config vocab_size={self.config.vocab_size} does not match "
                f"embedding rows={old_input_size}."
            )

        if target_vocab_size <= current_vocab_size:
            logger.info(f"Vocab size sufficient ({current_vocab_size} >= {target_vocab_size}), skipping resize.")
            return

        logger.info(f"Resizing vocab from {current_vocab_size} to {target_vocab_size}...")

        self.resize_token_embeddings(target_vocab_size, mean_resizing=False)

        if hasattr(self.config, "embedding_size"):
            self.config.embedding_size = target_vocab_size
        self.config.vocab_size = target_vocab_size
        self.config.extended_vocab_size = target_vocab_size
        self.model.config.vocab_size = target_vocab_size
        self.model.config.embedding_size = target_vocab_size

        wte = self.get_input_embeddings()
        ff_out = self.get_output_embeddings()

        with torch.no_grad():
            if init_strategy == "mean":
                existing_wte_data = wte.weight.data[:old_input_size]
                mean_embedding = existing_wte_data.mean(dim=0, keepdim=True)
                wte.weight.data[old_input_size:] = mean_embedding.expand(target_vocab_size - old_input_size, -1)
                logger.info(f"Initialized embedding layer with mean strategy (index {old_input_size}+).")
            else:
                wte.weight.data[old_input_size:].zero_()
                logger.info(f"Initialized embedding layer with zero strategy (index {old_input_size}+).")

            if not getattr(self.config, "weight_tying", False) and ff_out is not None:
                nn.init.normal_(
                    ff_out.weight.data[old_output_size:],
                    mean=0.0,
                    std=0.02,
                )
                logger.info(f"Initialized output projection weights with normal dist (index {old_output_size}+).")

        logger.info("Vocab resize and initialization complete.")

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.config.use_cache = False

        from .configuration_llada import ActivationCheckpointingStrategy

        if hasattr(self.model, "set_activation_checkpointing"):
            self.model.set_activation_checkpointing(
                ActivationCheckpointingStrategy.whole_layer
            )
            logger.info("Gestalt: Enabled Gradient Checkpointing (whole_layer strategy) via model method.")
        else:
            logger.error("Gestalt: Underlying model missing set_activation_checkpointing method!")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        input_embeddings = None
        adapter = getattr(self, "interaction_embedding_adapter", None)
        if adapter is not None and input_ids is not None:
            raw_embeds = self.model.transformer.wte(input_ids)
            inter_mask = (input_ids >= INTERACTION_START) & (input_ids < INTERACTION_END)
            if inter_mask.any():
                local_ids = (input_ids - INTERACTION_START).clamp(
                    min=0,
                    max=INTERACTION_END - INTERACTION_START - 1,
                )
                inter_embeds = adapter(local_ids)
                input_embeddings = torch.where(
                    inter_mask.unsqueeze(-1),
                    inter_embeds,
                    raw_embeds,
                )
            else:
                input_embeddings = raw_embeds

        outputs = self.model(
            input_ids=input_ids,
            input_embeddings=input_embeddings,
            attention_bias=None,
            **kwargs
        )

        return outputs, None, input_ids
    
    @torch.no_grad()
    def generate_mmu(
        self,
        image_tokens: List[int],
        token_h: int,
        token_w: int,
        system_tokens: List[int] = None,
        question_tokens: List[int] = None,
        gen_length: int = 128,
        steps: int = 64,
        block_length: int = 64,
        temperature: float = 0.0,
        remasking: str = "low_confidence"
    ):
        """Generate an MMU answer with block-wise diffusion decoding."""
        if token_h <= 0 or token_w <= 0:
            raise ValueError("token_h and token_w must be positive.")
        if len(image_tokens) != token_h * token_w:
            raise ValueError(
                f"Expected {token_h * token_w} image tokens, got {len(image_tokens)}."
            )
        if not all(
            SPECIAL_TOKENS.VISUAL_TOKEN_OFFSET <= token < SPECIAL_TOKENS.VISUAL_TOKEN_END
            for token in image_tokens
        ):
            raise ValueError("image_tokens must contain model-space visual token IDs.")
        if gen_length <= 0 or steps <= 0 or block_length <= 0:
            raise ValueError("gen_length, steps, and block_length must be positive.")
        if temperature < 0:
            raise ValueError("temperature must be non-negative.")
        if remasking not in {"low_confidence", "random"}:
            raise ValueError("remasking must be 'low_confidence' or 'random'.")

        device = self.model.transformer.wte.weight.device

        if system_tokens is None:
            system_tokens = []
        if question_tokens is None:
            question_tokens = []

        img_with_nl = list(image_tokens)
        prefix = build_mmu_condition(
            system_tokens,
            [img_with_nl],
            question_tokens,
        ) + [BOA, BOS]
        ans_start = len(prefix)
        ans_end = ans_start + gen_length

        token_list = prefix + [MASK] * gen_length + [EOA]
        x = torch.tensor(token_list, device=device).unsqueeze(0)
        document_ids = torch.zeros_like(x)

        generation_region = torch.zeros_like(x, dtype=torch.bool)
        generation_region[:, ans_start:ans_end] = True
        num_blocks = max(1, math.ceil(gen_length / block_length))
        if steps < num_blocks:
            raise ValueError(
                f"steps ({steps}) must be at least the number of blocks ({num_blocks})."
            )
        base_steps, extra_steps = divmod(steps, num_blocks)

        position_ids = compute_2d_position_ids(
            x,
            image_grids=[(int(token_h), int(token_w))],
        ).to(x.device)

        for block_idx in range(num_blocks):
            blk_s = ans_start + block_idx * block_length
            blk_e = min(ans_start + (block_idx + 1) * block_length, ans_end)
            block_mask = (x[:, blk_s:blk_e] == MASK)
            steps_per_block = base_steps + (1 if block_idx < extra_steps else 0)
            num_transfer_tokens = self._get_num_transfer_tokens(block_mask, steps_per_block)

            for step_i in range(steps_per_block):
                mask_index = (x == MASK) & generation_region
                logits = self.model(
                    input_ids=x,
                    position_ids=position_ids,
                    document_ids=document_ids,
                ).logits

                x0 = torch.argmax(self._add_gumbel_noise(logits, temperature), dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                else:
                    x0_p = torch.rand(x0.shape, device=device)

                x0_p[:, blk_e:] = -math.inf
                x0 = torch.where(mask_index, x0, x)
                conf = torch.where(mask_index, x0_p, torch.full_like(x0_p, -math.inf))

                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                k = int(num_transfer_tokens[0, step_i].item())
                if k > 0:
                    _, select_idx = torch.topk(conf[0], k=k)
                    transfer_index[0, select_idx] = True

                x[transfer_index] = x0[transfer_index]

            if (x[:, ans_start:ans_end] == EOS).any():
                break

        ans_tokens = x[0, ans_start:ans_end].tolist()
        if EOS in ans_tokens:
            ans_tokens = ans_tokens[:ans_tokens.index(EOS)]
        if MASK in ans_tokens:
            raise RuntimeError("MMU generation ended with unresolved mask tokens.")
        return ans_tokens

    def get_input_embeddings(self):
        return self.model.transformer.wte

    def set_input_embeddings(self, value):
        self.model.transformer.wte = value

    def get_output_embeddings(self):
        return getattr(self.model.transformer, "ff_out", None)

    def set_output_embeddings(self, value):
        self.model.transformer.ff_out = value

    def _cosine_schedule(self, t):
        return torch.cos(0.5 * math.pi * t)

    def _get_num_transfer_tokens(self, mask_index, steps):
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.long) + base
        for i in range(mask_num.size(0)):
            num_transfer[i, :remainder[i]] += 1
        return num_transfer
    
    
    def _gumbel_noise(self, t):
        u = torch.rand_like(t)
        return -torch.log(-torch.log(u.clamp(min=1e-20)) + 1e-20)
    
    def _gumbel_max_sample(self, logits, tau=1.0):
        if tau == 0.0:
            return logits.argmax(dim=-1)
        g = self._gumbel_noise(logits)
        return (logits / tau + g).argmax(dim=-1)

    def _add_gumbel_noise(self, logits, temperature=1.0):
        if temperature <= 1e-6: return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel = (-torch.log(noise + 1e-20)) ** temperature
        return logits.exp() / (gumbel + 1e-20)

    def _mask_by_random_topk(self, mask_len, probs, temperature=1.0):
        """Select exactly ``mask_len`` low-confidence tokens per batch row."""
        g = -torch.log(-torch.log(torch.rand_like(probs) + 1e-20) + 1e-20)
        confidence = torch.log(probs.clamp_min(1e-20)) + temperature * g
        ascending_indices = torch.argsort(confidence, dim=-1)
        selected = torch.zeros_like(probs, dtype=torch.bool)
        for batch_index in range(probs.shape[0]):
            count = int(mask_len[batch_index].item())
            count = max(0, min(count, probs.shape[1]))
            selected[batch_index, ascending_indices[batch_index, :count]] = True
        return selected
    
    @torch.no_grad()
    def generate_t2i(
        self,
        system_prompt_tokens: List[int],
        user_prompt_tokens: List[int],
        lat_h: int = 32,
        lat_w: int = 32,
        timesteps: int = 64,
        temperature: float = 1.0,
        cfg_scale: float = 4.0,
    ):
        """Generate visual tokens with the Stage-III T2I template."""
        if lat_h <= 0 or lat_w <= 0 or timesteps <= 0:
            raise ValueError("lat_h, lat_w, and timesteps must be positive.")
        if temperature < 0:
            raise ValueError("temperature must be non-negative.")
        if cfg_scale < 0:
            raise ValueError("cfg_scale must be non-negative.")
        VISUAL_OFFSET = VOCAB_CONFIG.VISUAL_TOKEN_OFFSET
        CODEBOOK_SIZE = VOCAB_CONFIG.VQVAE_CODEBOOK_SIZE

        device = self.model.transformer.wte.weight.device
        seq_len = lat_h * lat_w

        img_mask_with_nl = [MASK] * seq_len
        header = build_t2i_condition(
            system_prompt_tokens,
            user_prompt_tokens,
        ) + [BOA, BOI]
        tail = [EOI, EOA]
        token_list = header + img_mask_with_nl + tail

        x = torch.tensor(token_list, device=device).unsqueeze(0)
        document_ids = torch.zeros_like(x)

        position_ids = compute_2d_position_ids(
            x,
            image_grids=[(int(lat_h), int(lat_w))],
        ).to(x.device)

        img_start = len(header)
        img_end = img_start + len(img_mask_with_nl)

        generation_region = torch.zeros_like(x, dtype=torch.bool)
        generation_region[:, img_start:img_end] = True
        vq_mask = (x == MASK) & generation_region
        vq_len_total = vq_mask.sum(dim=1, keepdim=True)

        if cfg_scale > 0:
            uncond_header = build_t2i_condition(
                system_prompt_tokens,
                [],
            ) + [BOA, BOI]
            uncond_img_start = len(uncond_header)
            uncond_img_end = uncond_img_start + len(img_mask_with_nl)
        else:
            uncond_header = None
            uncond_img_start = uncond_img_end = 0

        for step in range(timesteps):
            if vq_mask.sum() == 0:
                break

            if step < timesteps - 1:
                frac = self._cosine_schedule(torch.tensor([(step + 1) / timesteps], device=device))
                keep_n = (vq_len_total.float() * frac).floor().clamp_min(1).long()
            else:
                keep_n = torch.zeros_like(vq_len_total)

            logits_cond = self.model(
                input_ids=x,
                position_ids=position_ids,
                document_ids=document_ids,
            ).logits
            vq_logits_cond = logits_cond[:, vq_mask[0], VISUAL_OFFSET : VISUAL_OFFSET + CODEBOOK_SIZE]

            if cfg_scale > 0:
                current_img_region = x[0, img_start:img_end].tolist()
                uncond_token_list = uncond_header + current_img_region + tail
                uncond_seq = torch.tensor(uncond_token_list, device=device).unsqueeze(0)

                position_ids_uncond = compute_2d_position_ids(
                    uncond_seq,
                    image_grids=[(int(lat_h), int(lat_w))],
                ).to(uncond_seq.device)

                uncond_out = self.model(
                    input_ids=uncond_seq,
                    position_ids=position_ids_uncond,
                    document_ids=torch.zeros_like(uncond_seq),
                ).logits
                uncond_vq_mask = torch.zeros_like(uncond_seq, dtype=torch.bool)
                uncond_vq_mask[:, uncond_img_start:uncond_img_end] = (
                    uncond_seq[:, uncond_img_start:uncond_img_end] == MASK
                )
                vq_logits_uncond = uncond_out[:, uncond_vq_mask[0], VISUAL_OFFSET : VISUAL_OFFSET + CODEBOOK_SIZE]

                vq_logits = (1 + cfg_scale) * vq_logits_cond - cfg_scale * vq_logits_uncond
            else:
                vq_logits = vq_logits_cond

            sampled = self._gumbel_max_sample(vq_logits, temperature)
            sampled_full = sampled + VISUAL_OFFSET

            probs = torch.softmax(vq_logits, dim=-1)
            conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

            flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]
            x.view(-1)[flat_idx] = sampled_full.view(-1)

            if keep_n > 0:
                mask_sel = self._mask_by_random_topk(
                    keep_n.squeeze(1), conf, temperature
                )
                x.view(-1)[flat_idx[mask_sel.view(-1)]] = MASK

            vq_mask = (x == MASK) & generation_region

        final_vq = x[0, img_start:img_end]
        if (final_vq == MASK).any():
            raise RuntimeError("T2I generation ended with unresolved mask tokens.")
        if not torch.all((final_vq >= VISUAL_OFFSET) & (final_vq < VISUAL_OFFSET + CODEBOOK_SIZE)):
            raise RuntimeError("T2I generation produced an out-of-range visual token.")

        return final_vq.unsqueeze(0)


from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register("gestalt", GestaltConfig)
AutoModelForCausalLM.register(GestaltConfig, GestaltModelLM)
