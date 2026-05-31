from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader, Dataset


DEFAULT_INPUT_KEYS = ("ego_history", "neighbor_history", "neighbor_mask", "risk_history")
DEFAULT_TARGET_KEYS = (
    "decision_label",
    "future_decision_sequence",
    "future_speed",
    "future_acceleration",
    "future_steer",
    "future_trajectory",
)
DEFAULT_META_KEYS = (
    "recording_id",
    "vehicle_id",
    "frame_id",
    "driving_direction",
    "current_lane_id",
    "lane_change_count",
    "return_flag",
    "time_to_first_lane_change",
)


class HighDIntentDataset(Dataset):
    """Lazy index-based Dataset for sharded highD intent samples.

    The split `.npz` stores shard paths plus row indices. This dataset opens
    shards lazily and keeps a small per-worker LRU cache of loaded npz handles.
    """

    def __init__(
        self,
        index_path: str | Path,
        input_keys: tuple[str, ...] = DEFAULT_INPUT_KEYS,
        target_keys: tuple[str, ...] = DEFAULT_TARGET_KEYS,
        meta_keys: tuple[str, ...] = DEFAULT_META_KEYS,
        include_meta: bool = True,
        cache_size: int = 3,
        preload_shards: bool = False,
    ) -> None:
        self.index_path = Path(index_path)
        if self.index_path.suffix.lower() in {".h5", ".hdf5"}:
            self._hdf5_dataset = HighDHDF5IntentDataset(
                self.index_path,
                input_keys=input_keys,
                target_keys=target_keys,
                meta_keys=meta_keys,
                include_meta=include_meta,
            )
            return
        self._hdf5_dataset = None
        index = np.load(self.index_path, allow_pickle=True)
        self.shards = np.asarray(index["shards"]).astype(str)
        self.recordings = np.asarray(index["recordings"]).astype(str)
        self.counts = np.asarray(index["counts"], dtype=np.int64)
        self.offsets = np.asarray(index["offsets"], dtype=np.int64)
        self.indices = np.asarray(index["indices"], dtype=np.int64)
        self.input_keys = input_keys
        self.target_keys = target_keys
        self.meta_keys = meta_keys
        self.include_meta = include_meta
        self.cache_size = int(cache_size)
        self.preload_shards = preload_shards
        self._cache: OrderedDict[int, Any] = OrderedDict()

        if self.offsets[-1] != len(self.indices):
            raise ValueError(f"Bad index file {self.index_path}: offsets do not match indices length.")

    def __len__(self) -> int:
        if self._hdf5_dataset is not None:
            return len(self._hdf5_dataset)
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> dict[str, Any]:
        if self._hdf5_dataset is not None:
            return self._hdf5_dataset[item]
        shard_id = int(np.searchsorted(self.offsets, item, side="right") - 1)
        local_offset = int(item - self.offsets[shard_id])
        sample_idx = int(self.indices[self.offsets[shard_id] + local_offset])
        shard = self._get_shard(shard_id)

        inputs = {key: _to_tensor(shard[key][sample_idx]) for key in self.input_keys}
        targets = {key: _to_tensor(shard[key][sample_idx]) for key in self.target_keys}
        sample: dict[str, Any] = {"inputs": inputs, "targets": targets}
        if self.include_meta:
            available = shard.keys() if isinstance(shard, dict) else shard.files
            meta = {key: _to_python_scalar(shard[key][sample_idx]) for key in self.meta_keys if key in available}
            meta["shard_id"] = shard_id
            meta["sample_index"] = sample_idx
            sample["meta"] = meta
        return sample

    def _get_shard(self, shard_id: int):
        if shard_id in self._cache:
            self._cache.move_to_end(shard_id)
            return self._cache[shard_id]
        npz = np.load(self.shards[shard_id], allow_pickle=True)
        if self.preload_shards:
            needed_keys = set(self.input_keys) | set(self.target_keys)
            if self.include_meta:
                needed_keys |= set(self.meta_keys)
            shard = {key: npz[key] for key in needed_keys if key in npz.files}
            npz.close()
        else:
            shard = npz
        self._cache[shard_id] = shard
        if len(self._cache) > self.cache_size:
            _, old = self._cache.popitem(last=False)
            if hasattr(old, "close"):
                old.close()
        return shard

    def close(self) -> None:
        if self._hdf5_dataset is not None:
            self._hdf5_dataset.close()
            return
        for shard in self._cache.values():
            if hasattr(shard, "close"):
                shard.close()
        self._cache.clear()


class HighDHDF5IntentDataset(Dataset):
    """Dataset backed by a packed HDF5 file."""

    def __init__(
        self,
        h5_path: str | Path,
        input_keys: tuple[str, ...] = DEFAULT_INPUT_KEYS,
        target_keys: tuple[str, ...] = DEFAULT_TARGET_KEYS,
        meta_keys: tuple[str, ...] = DEFAULT_META_KEYS,
        include_meta: bool = True,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.input_keys = input_keys
        self.target_keys = target_keys
        self.meta_keys = meta_keys
        self.include_meta = include_meta
        self._h5: h5py.File | None = None
        with h5py.File(self.h5_path, "r") as h5:
            self.length = int(h5.attrs.get("num_samples", h5["decision_label"].shape[0]))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item: int) -> dict[str, Any]:
        h5 = self._get_h5()
        inputs = {key: _to_tensor(h5[key][item]) for key in self.input_keys}
        targets = {key: _to_tensor(h5[key][item]) for key in self.target_keys}
        sample: dict[str, Any] = {"inputs": inputs, "targets": targets}
        if self.include_meta:
            sample["meta"] = {key: _to_python_scalar(h5[key][item]) for key in self.meta_keys if key in h5}
        return sample

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


def create_intent_dataloader(
    index_path: str | Path,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    include_meta: bool = False,
    pin_memory: bool = False,
    drop_last: bool = False,
    sampler=None,
    preload_shards: bool = False,
) -> DataLoader:
    dataset = HighDIntentDataset(index_path=index_path, include_meta=include_meta, preload_shards=preload_shards)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def _to_tensor(value: np.ndarray | np.generic) -> torch.Tensor:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.integer):
        return torch.as_tensor(array, dtype=torch.long)
    return torch.as_tensor(array, dtype=torch.float32)


def _to_python_scalar(value: np.ndarray | np.generic) -> int | float | str:
    if isinstance(value, np.generic):
        return value.item()
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return str(value)
