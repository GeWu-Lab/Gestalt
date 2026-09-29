import unittest

try:
    import torch
except ModuleNotFoundError:  # Lightweight source-check environments may omit PyTorch.
    torch = None

from gestalt.model.config import SPECIAL_TOKENS, VOCAB_CONFIG

if torch is not None:
    from gestalt.model.mrope import compute_2d_position_ids


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PositionIdTest(unittest.TestCase):
    def test_packed_documents_restart_positions(self):
        input_ids = torch.tensor([[10, 11, 12, 20, 21]])
        document_ids = torch.tensor([[0, 0, 0, 1, 1]])

        positions = compute_2d_position_ids(
            input_ids,
            document_ids=document_ids,
        )

        expected = torch.tensor([0, 1, 2, 0, 1])
        torch.testing.assert_close(positions[0, 0], expected)
        torch.testing.assert_close(positions[1, 0], expected)

    def test_image_uses_two_dimensional_grid(self):
        visual = VOCAB_CONFIG.VISUAL_TOKEN_OFFSET
        input_ids = torch.tensor([[
            10,
            SPECIAL_TOKENS.IMAGE_START,
            visual,
            visual + 1,
            visual + 2,
            visual + 3,
            SPECIAL_TOKENS.IMAGE_END,
            11,
        ]])

        positions = compute_2d_position_ids(
            input_ids,
            image_grids=[(2, 2)],
        )

        torch.testing.assert_close(
            positions[0, 0], torch.tensor([0, 1, 2, 2, 3, 3, 4, 5])
        )
        torch.testing.assert_close(
            positions[1, 0], torch.tensor([0, 1, 2, 3, 2, 3, 4, 5])
        )

    def test_grid_must_match_visual_token_count(self):
        visual = VOCAB_CONFIG.VISUAL_TOKEN_OFFSET
        input_ids = torch.tensor([[
            SPECIAL_TOKENS.IMAGE_START,
            visual,
            visual + 1,
            visual + 2,
            visual + 3,
            SPECIAL_TOKENS.IMAGE_END,
        ]])

        with self.assertRaisesRegex(ValueError, "does not match"):
            compute_2d_position_ids(
                input_ids,
                image_grids=[(1, 3)],
            )


if __name__ == "__main__":
    unittest.main()
