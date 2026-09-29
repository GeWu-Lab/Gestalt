import json
import os
import tempfile
import unittest
from unittest.mock import patch

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
except ModuleNotFoundError:  # Lightweight source-check environments may omit runtime deps.
    pa = None
    pq = None
    torch = None

from gestalt.model.config import SPECIAL_TOKENS

if torch is not None and pa is not None:
    from gestalt.data.datasets import (
        DataCollatorForCausalLM,
        GestaltDataset,
        Stage3Dataset,
        TokenBudgetPackingDataset,
    )
    from gestalt.data.stage3 import Stage3Processor


class FakeTokenizer:
    def __call__(self, text, **_kwargs):
        return {"input_ids": [1000 + ord(char) for char in text]}


@unittest.skipIf(torch is None or pa is None, "PyTorch and PyArrow are required")
class DataPipelineTest(unittest.TestCase):
    def _stage3_dataset_without_files(self):
        dataset = object.__new__(Stage3Dataset)
        dataset.processor = Stage3Processor(
            FakeTokenizer(),
            max_length=256,
        )
        dataset.processor.seed_rng(7)
        dataset._stage3_event_counts = {}
        return dataset

    def test_unified_t2i_row_reaches_processor_and_grid_validation(self):
        dataset = self._stage3_dataset_without_files()
        item = {
            "task_type": "t2i",
            "system_prompt": "",
            "user_prompt": "cat",
            "answer_image_tokens": [0, 1, 2, 3],
            "metadata_json": json.dumps({"token_height": 2, "token_width": 2}),
            "__data_source__": "fixture.parquet | Row: 0",
        }

        sample = dataset._process_item(item)

        self.assertEqual(sample["image_grids"], [(2, 2)])
        self.assertEqual(
            sum(token == SPECIAL_TOKENS.IMAGE_START for token in sample["inputs_id"]),
            1,
        )

    def test_unified_grid_mismatch_fails_fast(self):
        dataset = self._stage3_dataset_without_files()
        item = {
            "task_type": "t2i",
            "system_prompt": "",
            "user_prompt": "cat",
            "answer_image_tokens": [0, 1, 2, 3],
            "metadata_json": json.dumps({"token_height": 1, "token_width": 3}),
            "__data_source__": "fixture.parquet | Row: 1",
        }

        with self.assertRaisesRegex(RuntimeError, "grid 1x3 does not match 4"):
            dataset._process_item(item)

    def test_unknown_unified_task_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "Unsupported task_type"):
            Stage3Dataset._unified_to_processor_dict({
                "task_type": "audio",
                "metadata_json": "{}",
            })

    def test_legacy_row_is_rejected(self):
        dataset = self._stage3_dataset_without_files()
        with self.assertRaisesRegex(RuntimeError, "metadata_json"):
            dataset._process_item({
                "conversations": "[]",
                "img_tokens": [0],
                "metadata": {"token_height": 1, "token_width": 1},
                "__data_source__": "legacy.parquet | Row: 0",
            })

    def test_packing_and_collator_reset_document_positions(self):
        samples = [
            {
                "masked_inputs_id": [10, 11],
                "labels": [-100, 11],
                "loss_mask": [False, True],
                "ce_loss_weights": [0.0, 1.0],
                "image_grids": [],
            },
            {
                "masked_inputs_id": [20, 21, 22],
                "labels": [-100, -100, 22],
                "loss_mask": [False, False, True],
                "ce_loss_weights": [0.0, 0.0, 1.0],
                "image_grids": [],
            },
        ]
        packed = list(TokenBudgetPackingDataset(samples, max_length=5))
        self.assertEqual(len(packed), 1)
        self.assertEqual(packed[0]["document_ids"], [0, 0, 1, 1, 1])

        batch = DataCollatorForCausalLM()(packed)
        expected = torch.tensor([0, 1, 0, 1, 2])
        torch.testing.assert_close(batch["position_ids"][0, 0], expected)
        torch.testing.assert_close(batch["position_ids"][1, 0], expected)

    def test_labeled_token_requires_positive_weight(self):
        sample = {
            "masked_inputs_id": [SPECIAL_TOKENS.MASK],
            "labels": [42],
            "loss_mask": [True],
            "ce_loss_weights": [0.0],
        }
        with self.assertRaisesRegex(ValueError, "without positive"):
            list(TokenBudgetPackingDataset([sample], max_length=1))

    def test_empty_distributed_file_assignment_fails_before_iteration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = os.path.join(temp_dir, "one.parquet")
            pq.write_table(pa.table({"value": [1]}), parquet_path)
            dataset = GestaltDataset(parquet_path)

            with patch.dict(os.environ, {"RANK": "1", "WORLD_SIZE": "2"}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "No parquet shard assigned"):
                    next(iter(dataset))


if __name__ == "__main__":
    unittest.main()
