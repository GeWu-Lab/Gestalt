import unittest

from gestalt.data.stage3 import Stage3Processor
from gestalt.model.config import SPECIAL_TOKENS, VOCAB_CONFIG
from gestalt.model.templates import (
    DEFAULT_MMU_SYSTEM_PROMPT,
    build_i2i_condition,
    build_mmu_condition,
    build_t2i_condition,
)


class FakeTokenizer:
    """Small deterministic tokenizer for processor contract tests."""

    def __call__(self, text, **_kwargs):
        return {"input_ids": [1000 + ord(char) for char in text]}


class Stage3ProcessorTest(unittest.TestCase):
    def setUp(self):
        self.processor = Stage3Processor(
            tokenizer=FakeTokenizer(),
            max_length=256,
        )
        self.processor.seed_rng(7)

    def assert_training_contract(self, sample):
        fields = (
            "inputs_id",
            "masked_inputs_id",
            "loss_mask",
            "labels",
            "ce_loss_weights",
        )
        lengths = {field: len(sample[field]) for field in fields}
        self.assertEqual(len(set(lengths.values())), 1, lengths)
        self.assertTrue(any(sample["loss_mask"]))

        for original, masked, loss, label, weight in zip(
            sample["inputs_id"],
            sample["masked_inputs_id"],
            sample["loss_mask"],
            sample["labels"],
            sample["ce_loss_weights"],
        ):
            self.assertEqual(label, original if loss else -100)
            self.assertEqual(masked, SPECIAL_TOKENS.MASK if loss else original)
            self.assertEqual(weight > 0, loss)

    def test_i2t_contract(self):
        samples = self.processor({
            "conversations": [
                {"from": "human", "value": "<image>what?"},
                {"from": "gpt", "value": "cat"},
            ],
            "img_tokens": [0, 1, 2, 3],
            "metadata": {"token_height": 2, "token_width": 2},
        })
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assert_training_contract(sample)
        answer_start = sample["inputs_id"].index(SPECIAL_TOKENS.ANSWER_START)
        expected_condition = build_mmu_condition(
            FakeTokenizer()(DEFAULT_MMU_SYSTEM_PROMPT)["input_ids"],
            [[VOCAB_CONFIG.VISUAL_TOKEN_OFFSET + index for index in range(4)]],
            FakeTokenizer()("what?")["input_ids"],
        )
        self.assertEqual(sample["inputs_id"][:answer_start], expected_condition)
        for label in sample["labels"]:
            if label != -100:
                self.assertFalse(
                    VOCAB_CONFIG.VISUAL_TOKEN_OFFSET
                    <= label
                    < VOCAB_CONFIG.VISUAL_TOKEN_END
                )

    def test_t2i_contract(self):
        sample = self.processor({
            "system_prompt": "",
            "user_prompt": "cat",
            "answer_image": {"img_tokens": [0, 1, 2, 3]},
            "metadata": {"token_height": 2, "token_width": 2},
        })
        self.assert_training_contract(sample)
        answer_start = sample["inputs_id"].index(SPECIAL_TOKENS.ANSWER_START)
        self.assertEqual(
            sample["inputs_id"][:answer_start],
            build_t2i_condition([], FakeTokenizer()("cat")["input_ids"]),
        )
        for label in sample["labels"]:
            if label != -100:
                self.assertTrue(
                    VOCAB_CONFIG.VISUAL_TOKEN_OFFSET
                    <= label
                    < VOCAB_CONFIG.VISUAL_TOKEN_END
                )

    def test_i2i_contract(self):
        sample = self.processor({
            "system_prompt": "",
            "user_prompt": "edit",
            "input_image": {"img_tokens": [0, 1, 2, 3]},
            "answer_image": {"img_tokens": [4, 5, 6, 7]},
            "metadata": {
                "input_token_height": 2,
                "input_token_width": 2,
                "output_token_height": 2,
                "output_token_width": 2,
            },
        })
        self.assert_training_contract(sample)
        answer_start = sample["inputs_id"].index(SPECIAL_TOKENS.ANSWER_START)
        self.assertEqual(
            sample["inputs_id"][:answer_start],
            build_i2i_condition(
                [],
                [VOCAB_CONFIG.VISUAL_TOKEN_OFFSET + index for index in range(4)],
                FakeTokenizer()("edit")["input_ids"],
            ),
        )
        masked_image_tokens = sum(
            VOCAB_CONFIG.VISUAL_TOKEN_OFFSET
            <= label
            < VOCAB_CONFIG.VISUAL_TOKEN_END
            for label in sample["labels"]
            if label != -100
        )
        self.assertGreater(masked_image_tokens, 0)

    def test_all_i2t_rounds_are_preserved(self):
        conversations = []
        for index in range(3):
            conversations.extend([
                {"from": "human", "value": f"question {index}"},
                {"from": "gpt", "value": f"answer {index}"},
            ])
        samples = self.processor({
            "conversations": conversations,
            "img_tokens": [0, 1, 2, 3],
            "metadata": {"token_height": 2, "token_width": 2},
        })
        self.assertEqual(len(samples), 3)

    def test_mixed_image_token_spaces_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must contain either raw VQVAE"):
            self.processor({
                "system_prompt": "",
                "user_prompt": "cat",
                "answer_image": {
                    "img_tokens": [0, VOCAB_CONFIG.VISUAL_TOKEN_OFFSET]
                },
                "metadata": {"token_height": 1, "token_width": 2},
            })

    def test_missing_target_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no answer image tokens"):
            self.processor({
                "system_prompt": "",
                "user_prompt": "cat",
                "answer_image": {"img_tokens": []},
                "metadata": {"token_height": 2, "token_width": 2},
            })


if __name__ == "__main__":
    unittest.main()
