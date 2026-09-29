"""Stage-III parquet datasets and packing utilities."""

import os
import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import pyarrow.parquet as pq
import torch.distributed as dist
from torch.utils.data import IterableDataset
class GestaltDataset(IterableDataset):
    """Distributed, worker-aware parquet streaming dataset."""

    def __init__(
        self,
        data_path: Union[str, Path, List[str]],
        processor: Optional[Callable] = None,
        shuffle_buffer: int = 10000,
    ):
        self.data_paths = self._parse_data_path_arg(data_path)
        self.processor = processor
        if not isinstance(shuffle_buffer, int) or shuffle_buffer <= 0:
            raise ValueError("shuffle_buffer must be a positive integer.")
        self.shuffle_buffer = shuffle_buffer

        self.parquet_files = self._get_parquet_files()
        self._length = None
        self._file_row_counts = None

    @staticmethod
    def _parse_data_path_arg(data_path: Union[str, Path, List[str]]) -> List[Path]:
        """Parse data_path argument into a list of Path objects.

        Supports:
          - Single path string or Path:  "/some/dir"
          - Bracket-delimited multi-path string: "[/path/a, /path/b]"
          - Python list of strings: ["/path/a", "/path/b"]
        """
        if isinstance(data_path, (list, tuple)):
            paths = [str(path).strip() for path in data_path]
            return [Path(path) for path in paths if path]

        data_path_str = str(data_path).strip()
        if data_path_str.startswith("[") and data_path_str.endswith("]"):
            inner = data_path_str[1:-1]
            parts = [p.strip() for p in inner.split(",") if p.strip()]
            return [Path(p) for p in parts]

        return [Path(data_path_str)]

    def _get_parquet_files(self) -> List[Path]:
        all_files = []
        for dp in self.data_paths:
            all_files.extend(self._get_parquet_files_from_path(dp))
        if not all_files:
            raise FileNotFoundError(
                f"No parquet files found in: {[str(p) for p in self.data_paths]}"
            )
        print(f"[Dataset] Found {len(all_files)} parquet files from "
              f"{len(self.data_paths)} path(s)", flush=True)
        return sorted(all_files)

    @staticmethod
    def _get_parquet_files_from_path(data_path: Path) -> List[Path]:
        if data_path.is_dir():
            return list(data_path.rglob("*.parquet"))
        elif data_path.suffix == ".parquet" and data_path.is_file():
            return [data_path]
        elif data_path.suffix == ".parquet":
            raise FileNotFoundError(f"Parquet file does not exist: {data_path}")
        elif data_path.suffix in [".yaml", ".yml", ".json"]:
            return GestaltDataset._parse_config_file(data_path)
        else:
            raise ValueError(f"Unsupported path: {data_path}")

    @staticmethod
    def _parse_config_file(config_path: Path) -> List[Path]:
        import yaml, json

        with open(config_path) as f:
            if config_path.suffix in [".yaml", ".yml"]:
                cfg = yaml.safe_load(f)
            else:
                cfg = json.load(f)

        paths = []
        def _collect(obj):
            if isinstance(obj, str):
                p = Path(obj)
                if p.suffix == ".parquet":
                    paths.append(p)
                elif p.is_dir():
                    paths.extend(p.rglob("*.parquet"))
            elif isinstance(obj, list):
                for x in obj:
                    _collect(x)
            elif isinstance(obj, dict):
                for v in obj.values():
                    _collect(v)

        _collect(cfg)
        return sorted(paths)

    def _get_file_row_counts(self) -> List[int]:
        if self._file_row_counts is None:
            self._file_row_counts = [
                pq.ParquetFile(path).metadata.num_rows for path in self.parquet_files
            ]
        return self._file_row_counts

    def _build_weighted_file_assignments(self, total_workers: int) -> List[List[int]]:
        import heapq

        if total_workers <= 0:
            raise ValueError(f"total_workers must be positive, got {total_workers}")

        row_counts = self._get_file_row_counts()
        assignments = [[] for _ in range(total_workers)]
        heap = [(0, worker_id) for worker_id in range(total_workers)]
        heapq.heapify(heap)

        # Largest Processing Time first: assign the heaviest file to the lightest worker.
        weighted_files = sorted(
            enumerate(row_counts),
            key=lambda x: (-x[1], x[0]),
        )

        for file_idx, row_count in weighted_files:
            assigned_rows, worker_id = heapq.heappop(heap)
            assignments[worker_id].append(file_idx)
            heapq.heappush(heap, (assigned_rows + row_count, worker_id))

        return assignments

    def _iter_parquet_files(self, file_indices: List[int] = None,
                            global_worker_id: int = 0,
                            epoch: int = 0) -> Iterator[Dict]:
        files_to_read = self.parquet_files

        if file_indices is not None:
            files_to_read = [self.parquet_files[i] for i in file_indices]

        if not files_to_read:
            return

        for file_path in files_to_read:
            try:
                pf = pq.ParquetFile(file_path)
                row_offset = 0
                for batch in pf.iter_batches(batch_size=1000):
                    batch_dict = batch.to_pydict()
                    n_rows = len(next(iter(batch_dict.values())))

                    for i in range(n_rows):
                        item = {k: v[i] for k, v in batch_dict.items()}
                        item["__data_source__"] = f"{file_path.name} | Row: {row_offset + i}"
                        yield item

                    row_offset += n_rows

            except Exception as e:
                worker_info = torch.utils.data.get_worker_info()
                worker_id = worker_info.id if worker_info is not None else 0
                raise RuntimeError(
                    f"Worker {worker_id} failed while reading {file_path}: {e}"
                ) from e

    def __iter__(self) -> Iterator[Dict[str, Any]]:

        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        total_workers = world_size * num_workers
        global_worker_id = rank * num_workers + worker_id

        # Weighted file assignment by row count; each file goes to exactly one worker.
        assignments = self._build_weighted_file_assignments(total_workers)
        my_files_indices = assignments[global_worker_id]

        # Every distributed worker must participate in the same training steps.
        if not my_files_indices:
            raise RuntimeError(
                f"No parquet shard assigned to global worker {global_worker_id}. "
                f"Found {len(self.parquet_files)} files for {total_workers} workers; "
                "reduce WORLD_SIZE/dataloader_num_workers or provide more shards."
            )

        # Seed the processor's RNG with global_worker_id so each rank/worker
        # gets an independent mask-ratio sequence.
        if self.processor is not None and hasattr(self.processor, 'seed_rng'):
            seed = (torch.initial_seed() + global_worker_id) % (2**32 - 1)
            self.processor.seed_rng(seed)

        epoch = 0
        while True:
            iterator = self._iter_parquet_files(file_indices=my_files_indices,
                                                global_worker_id=global_worker_id,
                                                epoch=epoch)
            iterator = self._shuffle_iterator(
                iterator,
                seed=torch.initial_seed() + global_worker_id + epoch * 1000003,
            )

            yielded_any = False
            for item in iterator:
                processed = self._process_item(item)
                if processed is not None:
                    if isinstance(processed, list):
                        for sample in processed:
                            if sample is not None:
                                yielded_any = True
                                yield sample
                    else:
                        yielded_any = True
                        yield processed

            if not yielded_any:
                break
            epoch += 1

    def _shuffle_iterator(self, iterator: Iterator, seed: int) -> Iterator:
        """Bounded-memory deterministic shuffle for one rank/worker epoch."""
        rng = random.Random(seed)
        buffer = []
        for item in iterator:
            buffer.append(item)
            if len(buffer) >= self.shuffle_buffer:
                index = rng.randrange(len(buffer))
                yield buffer[index]
                buffer[index] = buffer[-1]
                buffer.pop()
        rng.shuffle(buffer)
        yield from buffer

    def _process_item(self, item: Dict) -> Optional[Dict]:

        if self.processor:
            return self.processor(item)
        return item

    def __len__(self) -> int:
        if self._length is None:
            self._length = sum(self._get_file_row_counts())
        return self._length


class Stage3Dataset(GestaltDataset):
    """Stage-III streaming dataset for the unified Parquet schema only."""

    def __init__(
        self,
        data_path: str,
        processor=None,
        shuffle_buffer: int = 10000,
    ):
        super().__init__(data_path, processor, shuffle_buffer=shuffle_buffer)
        self._stage3_event_counts = {}

    @staticmethod
    def _has_tokens(tokens) -> bool:
        """Return True for non-empty token sequences without numpy truthiness."""
        if tokens is None:
            return False
        try:
            return len(tokens) > 0
        except TypeError:
            return False

    @staticmethod
    def _normalize_unified_task_type(item: Dict) -> str:
        """Validate the three canonical unified-schema task names."""
        if "task_type" not in item:
            raise ValueError("Unified row is missing required field 'task_type'.")
        task_type = str(item["task_type"]).strip().lower()
        if task_type not in {"i2t", "t2i", "i2i"}:
            raise ValueError(
                f"Unsupported task_type={item.get('task_type')!r}; "
                "expected 'i2t', 't2i', or 'i2i'."
            )
        return task_type

    @staticmethod
    def _unified_to_processor_dict(item: Dict) -> Dict:
        """Convert unified schema row to the dict format Stage3Processor expects.

        Builds the processor's compact per-task structure:
          i2t      → {conversations, img_tokens, metadata}
          t2i      → {system_prompt, user_prompt, answer_image: {img_tokens}, metadata}
          i2i      → {system_prompt, user_prompt, input_image: {img_tokens},
                       answer_image: {img_tokens}, metadata}
        """
        import json as _json

        if "metadata_json" not in item:
            raise ValueError("Unified row is missing required field 'metadata_json'.")
        task_type = Stage3Dataset._normalize_unified_task_type(item)
        metadata_raw = item.get("metadata_json")
        if metadata_raw is None or metadata_raw == "":
            metadata = {}
        elif isinstance(metadata_raw, str):
            metadata = _json.loads(metadata_raw)
        else:
            raise ValueError("metadata_json must be a JSON string or null.")
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json must decode to a JSON object.")

        result = {"metadata": metadata}

        if task_type == "i2t":
            result["conversations"] = item.get("conversations")
            result["img_tokens"] = item.get("img_tokens")
        elif task_type == "t2i":
            result["system_prompt"] = item.get("system_prompt", "")
            result["user_prompt"] = item.get("user_prompt", "")
            answer_tokens = item.get("answer_image_tokens")
            result["answer_image"] = (
                {"img_tokens": answer_tokens}
                if Stage3Dataset._has_tokens(answer_tokens)
                else {}
            )
        elif task_type == "i2i":
            result["system_prompt"] = item.get("system_prompt", "")
            result["user_prompt"] = item.get("user_prompt", "")
            input_tokens = item.get("input_image_tokens")
            answer_tokens = item.get("answer_image_tokens")
            result["input_image"] = (
                {"img_tokens": input_tokens}
                if Stage3Dataset._has_tokens(input_tokens)
                else {}
            )
            result["answer_image"] = (
                {"img_tokens": answer_tokens}
                if Stage3Dataset._has_tokens(answer_tokens)
                else {}
            )
        return result

    @staticmethod
    def _event_key(item: Dict, event: str) -> Tuple[str, str, str]:
        task_type = str(item.get("task_type", "unknown"))
        dataset = str(item.get("__source_dataset__", "unknown"))
        return event, task_type, dataset

    def _record_event(self, event: str, item: Dict, source: str, detail: str = ""):
        """Log skipped/failed rows with bounded noise and per-worker counters."""
        key = self._event_key(item, event)
        count = self._stage3_event_counts.get(key, 0) + 1
        self._stage3_event_counts[key] = count

        if count <= 5 or count in {10, 50, 100} or count % 1000 == 0:
            task_type = key[1]
            dataset = key[2]
            detail_msg = f" | {detail}" if detail else ""
            print(
                f"[Stage3 {event}] count={count} | task_type={task_type} | "
                f"dataset={dataset} | {source}{detail_msg}",
                flush=True,
            )

    def _process_item(self, item: Dict) -> Optional[Dict]:
        source = item.get('__data_source__', 'Unknown Source')

        try:
            if self.processor is None:
                self._record_event("SKIP_NO_PROCESSOR", item, source)
                return None

            processor_input = self._unified_to_processor_dict(item)
            processor_input["__data_source__"] = source

            processed = self.processor(processor_input)

            if processed is None:
                processor_event = processor_input.get("__stage3_processor_event__")
                if processor_event:
                    self._record_event(
                        processor_event.get("event", "PROCESSOR_EVENT"),
                        item,
                        source,
                        detail=processor_event.get("detail", ""),
                    )
                else:
                    self._record_event("SKIP_PROCESSOR_NONE", item, source)
                return None

            metadata = processor_input.get("metadata", {})

            def _finalize_sample(
                sample: Dict,
                sample_idx: int = None,
            ) -> Optional[Dict]:
                if sample is None:
                    return None

                sample["__data_source__"] = (
                    f"{source} | split_qa:{sample_idx}" if sample_idx is not None else source
                )

                # Build image_grids for MRoPE
                in_h  = metadata.get("input_token_height")
                in_w  = metadata.get("input_token_width")
                out_h = metadata.get("output_token_height")
                out_w = metadata.get("output_token_width")
                if in_h and in_w and out_h and out_w:
                    sample["image_grids"] = [(int(in_h), int(in_w)), (int(out_h), int(out_w))]
                else:
                    token_h = metadata.get("token_height")
                    token_w = metadata.get("token_width")
                    if token_h and token_w:
                        sample["image_grids"] = [(int(token_h), int(token_w))]
                    else:
                        images = metadata.get("images")
                        if images and len(images) > 0:
                            grids = []
                            for img_info in images:
                                th = img_info.get("token_height")
                                tw = img_info.get("token_width")
                                if th and tw:
                                    grids.append((int(th), int(tw)))
                            if grids:
                                sample["image_grids"] = grids

                from gestalt.model.config import SPECIAL_TOKENS
                token_ids = sample["inputs_id"]
                starts = [
                    index for index, token in enumerate(token_ids)
                    if token == SPECIAL_TOKENS.IMAGE_START
                ]
                ends = [
                    index for index, token in enumerate(token_ids)
                    if token == SPECIAL_TOKENS.IMAGE_END
                ]
                grids = sample.get("image_grids", [])
                if len(starts) != len(ends) or len(grids) != len(starts):
                    raise ValueError(
                        f"Expected one image grid per image region; found "
                        f"starts={len(starts)}, ends={len(ends)}, grids={len(grids)}."
                    )
                for image_index, (start, end, grid) in enumerate(zip(starts, ends, grids)):
                    height, width = map(int, grid)
                    token_count = end - start - 1
                    if end <= start or height <= 0 or width <= 0 or height * width != token_count:
                        raise ValueError(
                            f"Image {image_index} grid {height}x{width} does not match "
                            f"{token_count} image tokens."
                        )

                return sample

            if isinstance(processed, list):
                finalized = []
                for idx, sample in enumerate(processed):
                    sample = _finalize_sample(sample, idx)
                    if sample is not None:
                        finalized.append(sample)
                if not finalized:
                    self._record_event("SKIP_FINALIZE_EMPTY", item, source)
                    return None

                return finalized

            return _finalize_sample(processed)

        except Exception as e:
            detail = f"conversations={str(item.get('conversations', 'N/A'))[:100]}... | error={e}"
            self._record_event("ERROR_PROCESS_ITEM", item, source, detail=detail)
            raise RuntimeError(f"Failed to process {source}: {e}") from e

class TokenBudgetPackingDataset(IterableDataset):
    """
    Wraps a Stage3Dataset and accumulates samples until reaching a token budget.

    Each yielded item is a pre-packed dict containing multiple documents
    concatenated together, with document_ids tracking boundaries.
    Designed to be used with batch_size=1 in the Trainer.
    """

    def __init__(self, dataset: IterableDataset, max_length: int = 10000):
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("Packing max_length must be a positive integer.")
        self.dataset = dataset
        self.max_length = max_length

    def _new_bin(self) -> Dict:
        return {
            "masked_inputs_id": [],
            "labels": [],
            "loss_mask": [],
            "ce_loss_weights": [],
            "document_ids": [],
            "image_grids": [],
            "num_docs": 0,
        }

    @staticmethod
    def _validate_training_fields(sample: Dict, context: str):
        """Validate token-level fields before packing them into a training bin."""
        required = ["masked_inputs_id", "labels", "loss_mask", "ce_loss_weights"]
        missing = [key for key in required if key not in sample]
        if missing:
            raise ValueError(f"{context}: missing required field(s): {missing}")

        lengths = {key: len(sample[key]) for key in required}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{context}: inconsistent token field lengths: {lengths}")

        for idx, (label, is_loss, weight) in enumerate(
            zip(sample["labels"], sample["loss_mask"], sample["ce_loss_weights"])
        ):
            has_label = label != -100
            if has_label != bool(is_loss):
                raise ValueError(
                    f"{context}: labels/loss_mask mismatch at token {idx}: "
                    f"label={label}, loss_mask={is_loss}"
                )
            if weight < 0:
                raise ValueError(f"{context}: negative ce_loss_weight at token {idx}: {weight}")
            if not math.isfinite(weight):
                raise ValueError(f"{context}: non-finite ce_loss_weight at token {idx}: {weight}")
            if weight > 0 and not has_label:
                raise ValueError(
                    f"{context}: positive ce_loss_weight without label at token {idx}: "
                    f"weight={weight}"
                )
            if has_label and weight <= 0:
                raise ValueError(
                    f"{context}: label without positive ce_loss_weight at token {idx}: "
                    f"label={label}, weight={weight}"
                )

    def __iter__(self) -> Iterator[Dict]:
        current = self._new_bin()
        current_len = 0

        for sample in self.dataset:
            if sample is None:
                print("[DATA SKIP] Empty sample, skipping", flush=True)
                continue

            source = sample.get("__data_source__", "unknown")
            self._validate_training_fields(sample, f"sample {source}")
            sample_ids = sample["masked_inputs_id"]
            sample_len = len(sample_ids)
            if sample_len == 0:
                continue

            # Processor and packing limits must agree; never create malformed
            # sequences by truncating after task construction.
            if sample_len > self.max_length:
                raise ValueError(
                    f"Sample {source} has {sample_len} tokens, exceeding the "
                    f"packing budget {self.max_length}."
                )

            # Would exceed budget: yield current bin, start new one
            if current_len + sample_len > self.max_length:
                result = self._emit_bin(current)
                if result is not None:
                    yield result
                current = self._new_bin()
                current_len = 0

            # Append sample to current bin
            doc_id = current["num_docs"]
            self._append_sample(current, sample, doc_id)
            current_len += sample_len

        # Yield remaining samples at end of epoch
        if current_len > 0:
            result = self._emit_bin(current)
            if result is not None:
                yield result

    def _emit_bin(self, bin_dict: Dict) -> Optional[Dict]:
        """Finalize and return a bin, or None if it has no positive loss weight."""
        if sum(bin_dict["ce_loss_weights"]) <= 0:
            return None
        self._validate_training_fields(bin_dict, "packed bin")
        return self._finalize(bin_dict)

    def _append_sample(self, bin_dict: Dict, sample: Dict, doc_id: int):
        sample_ids = sample["masked_inputs_id"]
        sample_len = len(sample_ids)
        bin_dict["masked_inputs_id"].extend(sample_ids)
        bin_dict["labels"].extend(sample["labels"])
        bin_dict["loss_mask"].extend(sample["loss_mask"])
        bin_dict["ce_loss_weights"].extend(sample["ce_loss_weights"])
        bin_dict["document_ids"].extend([doc_id] * sample_len)
        grids = sample.get("image_grids")
        if grids:
            bin_dict["image_grids"].extend(grids)
        bin_dict["num_docs"] += 1

    def _finalize(self, bin_dict: Dict) -> Dict:
        """Remove internal bookkeeping and return clean dict."""
        return {
            "masked_inputs_id": bin_dict["masked_inputs_id"],
            "labels": bin_dict["labels"],
            "loss_mask": bin_dict["loss_mask"],
            "ce_loss_weights": bin_dict["ce_loss_weights"],
            "document_ids": bin_dict["document_ids"],
            "image_grids": bin_dict["image_grids"],
        }

    def __len__(self):
        parent_len = len(self.dataset)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        per_rank_samples = parent_len // world_size
        estimated_batches = per_rank_samples * 1500 // self.max_length
        return max(1, estimated_batches)

class DataCollatorForCausalLM:
    """Collator for use with TokenBudgetPackingDataset.

    Expects batch_size=1 where each feature is a pre-packed dict from
    TokenBudgetPackingDataset. Converts lists to tensors and computes
    2D position IDs.
    """

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        from gestalt.model.mrope import compute_2d_position_ids

        # batch_size=1: single pre-packed feature from TokenBudgetPackingDataset
        if len(features) != 1:
            raise ValueError(
                f"DataCollatorForCausalLM expects batch_size=1 (got {len(features)}). "
                f"Set per_device_train_batch_size=1 when using TokenBudgetPackingDataset."
            )
        feat = features[0]

        input_ids_t = torch.tensor([feat["masked_inputs_id"]], dtype=torch.long)  # (1, L)
        labels_t = torch.tensor([feat["labels"]], dtype=torch.long)
        document_ids_t = torch.tensor([feat["document_ids"]], dtype=torch.long)
        loss_mask_t = torch.tensor([feat["loss_mask"]], dtype=torch.bool)
        ce_loss_weights_t = torch.tensor([feat["ce_loss_weights"]], dtype=torch.float32)

        pos_2d = compute_2d_position_ids(
            input_ids_t,
            image_grids=feat["image_grids"],
            document_ids=document_ids_t,
        )  # (2, 1, L)

        result = {
            "input_ids": input_ids_t,
            "labels": labels_t,
            "position_ids": pos_2d,
            "document_ids": document_ids_t,
            "loss_mask": loss_mask_t,
            "ce_loss_weights": ce_loss_weights_t,
        }

        return result
