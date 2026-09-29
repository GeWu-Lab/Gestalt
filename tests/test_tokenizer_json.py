import json
import os
from pathlib import Path
import unittest

from gestalt.model.config import SPECIAL_TOKENS, VOCAB_CONFIG
from gestalt.model.builder import load_tokenizer


class TokenizerJsonTest(unittest.TestCase):
    def test_tokenizer_ids(self):
        configured = os.environ.get("GESTALT_TOKENIZER")
        path = Path(configured) if configured else Path(__file__).parents[1] / "tokenizer.json"
        if path.is_dir():
            path = path / "tokenizer.json"
        if not path.is_file():
            self.skipTest("Set GESTALT_TOKENIZER to a checkpoint or tokenizer.json")

        data = json.loads(path.read_text(encoding="utf-8"))
        token_to_id = {
            item["content"]: item["id"]
            for item in data.get("added_tokens", [])
        }
        expected = {
            "<|startoftext|>": SPECIAL_TOKENS.BOS,
            "<|endoftext|>": SPECIAL_TOKENS.EOS,
            "<system>": SPECIAL_TOKENS.SYSTEM_START,
            "</system>": SPECIAL_TOKENS.SYSTEM_END,
            "<user>": SPECIAL_TOKENS.USER_START,
            "</user>": SPECIAL_TOKENS.USER_END,
            "<|mdm_mask|>": SPECIAL_TOKENS.MASK,
            "<padding>": SPECIAL_TOKENS.PADDING,
            "<IMAGE>": SPECIAL_TOKENS.IMAGE_START,
            "</IMAGE>": SPECIAL_TOKENS.IMAGE_END,
            "<answer>": SPECIAL_TOKENS.ANSWER_START,
            "</answer>": SPECIAL_TOKENS.ANSWER_END,
        }
        for token, expected_id in expected.items():
            self.assertEqual(token_to_id.get(token), expected_id, token)

        all_ids = list(data["model"]["vocab"].values()) + [
            item["id"] for item in data.get("added_tokens", [])
        ]
        self.assertEqual(max(all_ids) + 1, VOCAB_CONFIG.VISUAL_TOKEN_END)

    def test_loader_accepts_checkpoint_directory(self):
        configured = os.environ.get("GESTALT_TOKENIZER")
        if not configured:
            self.skipTest("Set GESTALT_TOKENIZER to a checkpoint or tokenizer.json")
        source = Path(configured)
        checkpoint = source if source.is_dir() else source.parent
        if not (checkpoint / "tokenizer.json").is_file():
            self.skipTest("Configured checkpoint has no tokenizer.json")

        tokenizer = load_tokenizer(str(checkpoint))

        self.assertEqual(len(tokenizer), VOCAB_CONFIG.VISUAL_TOKEN_END)
        self.assertEqual(tokenizer.bos_token_id, SPECIAL_TOKENS.BOS)
        self.assertEqual(tokenizer.eos_token_id, SPECIAL_TOKENS.EOS)
        self.assertEqual(tokenizer.pad_token_id, SPECIAL_TOKENS.PADDING)


if __name__ == "__main__":
    unittest.main()
