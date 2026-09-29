import unittest
from types import SimpleNamespace

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # Lightweight source-check environments may omit PyTorch.
    torch = None
    nn = None

from gestalt.model.config import SPECIAL_TOKENS, VOCAB_CONFIG

if torch is not None:
    from gestalt.model.modeling_gestalt import GestaltModelLM


class FakeCore(nn.Module if nn is not None else object):
    def __init__(self, preferred_token):
        super().__init__()
        holder = nn.Module()
        holder.wte = nn.Embedding(1, 1)
        self.transformer = holder
        self.preferred_token = preferred_token
        self.seen_inputs = []

    def forward(self, input_ids, **_kwargs):
        self.seen_inputs.append(input_ids.detach().clone())
        logits = torch.zeros(
            (*input_ids.shape, VOCAB_CONFIG.VISUAL_TOKEN_END),
            device=input_ids.device,
        )
        logits[..., self.preferred_token] = 5.0
        return SimpleNamespace(logits=logits)


@unittest.skipIf(torch is None, "PyTorch and Transformers are required")
class InferenceSamplingTest(unittest.TestCase):
    @staticmethod
    def _model(preferred_token):
        model = GestaltModelLM.__new__(GestaltModelLM)
        nn.Module.__init__(model)
        core = FakeCore(preferred_token)
        model.model = core
        return model, core

    def test_t2i_only_updates_answer_region_and_returns_visual_tokens(self):
        target = VOCAB_CONFIG.VISUAL_TOKEN_OFFSET + 7
        model, core = self._model(target)

        result = model.generate_t2i(
            system_prompt_tokens=[],
            user_prompt_tokens=[SPECIAL_TOKENS.MASK],
            lat_h=2,
            lat_w=2,
            timesteps=2,
            temperature=0.0,
            cfg_scale=0.0,
        )

        self.assertEqual(result.tolist(), [[target] * 4])
        self.assertEqual(len(core.seen_inputs), 2)
        # One prompt MASK remains untouched. The four answer MASKs become two
        # after the first scheduled remasking step, then zero after the last.
        self.assertEqual((core.seen_inputs[0] == SPECIAL_TOKENS.MASK).sum().item(), 5)
        self.assertEqual((core.seen_inputs[1] == SPECIAL_TOKENS.MASK).sum().item(), 3)

    def test_t2i_rejects_negative_cfg(self):
        model, _ = self._model(VOCAB_CONFIG.VISUAL_TOKEN_OFFSET)
        with self.assertRaisesRegex(ValueError, "cfg_scale"):
            model.generate_t2i([], [], lat_h=1, lat_w=1, cfg_scale=-1.0)

    def test_mmu_reports_unresolved_masks(self):
        model, _ = self._model(SPECIAL_TOKENS.MASK)
        image_tokens = [VOCAB_CONFIG.VISUAL_TOKEN_OFFSET]

        with self.assertRaisesRegex(RuntimeError, "unresolved mask"):
            model.generate_mmu(
                image_tokens=image_tokens,
                token_h=1,
                token_w=1,
                system_tokens=[],
                question_tokens=[],
                gen_length=1,
                steps=1,
                block_length=1,
                temperature=0.0,
            )

    def test_random_topk_selects_exact_count_even_for_ties(self):
        model, _ = self._model(VOCAB_CONFIG.VISUAL_TOKEN_OFFSET)
        selected = model._mask_by_random_topk(
            torch.tensor([2]),
            torch.ones((1, 4)),
            temperature=0.0,
        )
        self.assertEqual(selected.sum().item(), 2)


if __name__ == "__main__":
    unittest.main()
