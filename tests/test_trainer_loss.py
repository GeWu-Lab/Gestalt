import unittest
from types import SimpleNamespace

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import transformers  # noqa: F401
except ModuleNotFoundError:  # Lightweight source-check environments may omit runtime deps.
    torch = None
    F = None

from gestalt.model.config import SPECIAL_TOKENS, VOCAB_CONFIG

if torch is not None:
    from gestalt.model.builder import validate_model_vocab
    from gestalt.model.modeling_gestalt import GestaltConfig, GestaltModelLM
    from gestalt.training.trainer import GestaltTrainer
    from gestalt.training.trainer import _fold_interaction_adapter_into_wte


class FakeModel:
    def __init__(self, logits):
        self.logits = logits
        self.inputs = None

    def __call__(self, **inputs):
        self.inputs = inputs
        return SimpleNamespace(logits=self.logits), None, inputs["input_ids"]


@unittest.skipIf(torch is None, "PyTorch and Transformers are required")
class TrainerLossTest(unittest.TestCase):
    @staticmethod
    def _trainer():
        return SimpleNamespace(args=SimpleNamespace())

    @staticmethod
    def _inputs(labels, weights):
        return {
            "input_ids": torch.zeros_like(labels),
            "labels": labels,
            "position_ids": None,
            "document_ids": torch.zeros_like(labels),
            "loss_mask": labels != -100,
            "ce_loss_weights": weights,
        }

    def test_weighted_cross_entropy_and_return_contract(self):
        logits = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]]])
        labels = torch.tensor([[-100, 0, 0]])
        weights = torch.tensor([[0.0, 1.0, 3.0]])
        inputs = self._inputs(labels, weights)

        model = FakeModel(logits)
        loss, outputs = GestaltTrainer.compute_loss(
            self._trainer(),
            model,
            inputs,
            return_outputs=True,
        )

        token_losses = F.cross_entropy(
            logits.reshape(-1, 2), labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
        expected = (token_losses * weights.reshape(-1)).sum() / weights.sum()
        torch.testing.assert_close(loss, expected)
        self.assertIs(outputs.logits, logits)
        self.assertNotIn("labels", model.inputs)

    def test_zero_weight_for_supervised_label_is_rejected(self):
        logits = torch.zeros((1, 1, 2))
        inputs = self._inputs(
            torch.tensor([[1]]),
            torch.tensor([[0.0]]),
        )
        with self.assertRaisesRegex(ValueError, "positive CE loss weight"):
            GestaltTrainer.compute_loss(
                self._trainer(),
                FakeModel(logits),
                inputs,
            )

    def test_tiny_gestalt_forward_backward(self):
        config = GestaltConfig(
            vocab_size=128,
            embedding_size=128,
            extended_vocab_size=128,
            d_model=8,
            n_heads=2,
            n_layers=2,
            mlp_ratio=2,
            block_group_size=1,
            weight_tying=False,
            rope=True,
            inter_layer_num=1,
            attention_dropout=0.0,
            residual_dropout=0.0,
            embedding_dropout=0.0,
        )
        model = GestaltModelLM(config)
        model.model.reset_parameters()
        model.eval()  # CPU test uses the dense SDPA path.

        input_ids = torch.tensor([[10, 11, 12]])
        inputs = self._inputs(
            torch.tensor([[-100, 11, 12]]),
            torch.tensor([[0.0, 1.0, 1.0]]),
        )
        inputs["input_ids"] = input_ids

        loss = GestaltTrainer.compute_loss(self._trainer(), model, inputs)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        embedding_grad = model.get_input_embeddings().weight.grad
        self.assertIsNotNone(embedding_grad)
        self.assertGreater(embedding_grad.norm().item(), 0.0)

    def test_tiny_model_vocab_resize_preserves_and_initializes_rows(self):
        config = GestaltConfig(
            vocab_size=20,
            embedding_size=20,
            extended_vocab_size=24,
            d_model=8,
            n_heads=2,
            n_layers=2,
            mlp_ratio=2,
            block_group_size=1,
            weight_tying=False,
            rope=True,
            inter_layer_num=1,
        )
        model = GestaltModelLM(config)
        old_input = model.get_input_embeddings().weight.detach().clone()
        expected_mean = old_input.mean(dim=0)

        model.resize_and_initialize_vocab(24)

        torch.testing.assert_close(model.get_input_embeddings().weight[:20], old_input)
        torch.testing.assert_close(
            model.get_input_embeddings().weight[20:],
            expected_mean.expand(4, -1),
        )
        self.assertEqual(model.get_output_embeddings().weight.shape[0], 24)
        validate_model_vocab(model, 24)

    def test_interaction_adapter_is_folded_into_token_embeddings(self):
        class AdapterModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.wte = nn.Embedding(VOCAB_CONFIG.EXTENDED_VOCAB_SIZE, 2)
                self.interaction_embedding_adapter = nn.Embedding(32, 2)
                self.config = SimpleNamespace(use_interaction_embedding_adapter=True)

            def get_input_embeddings(self):
                return self.wte

        model = AdapterModel()
        expected = torch.arange(64, dtype=torch.float32).reshape(32, 2)
        with torch.no_grad():
            model.interaction_embedding_adapter.weight.copy_(expected)

        folded = _fold_interaction_adapter_into_wte(
            model,
            loading_info={"missing_keys": []},
            checkpoint_path="unused",
        )

        self.assertTrue(folded)
        start = SPECIAL_TOKENS.INTERACTION_TOKEN_START
        torch.testing.assert_close(model.wte.weight[start:start + 32], expected)
        self.assertFalse(hasattr(model, "interaction_embedding_adapter"))
        self.assertFalse(model.config.use_interaction_embedding_adapter)


if __name__ == "__main__":
    unittest.main()
