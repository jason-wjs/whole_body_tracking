"""Runtime motion source with multi-dataset clip sampling and clip-bin adaptive sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from whole_body_tracking.data_process.compiled_dataset import CompiledMotionDataset


@dataclass
class _DatasetAdaptiveState:
    clip_bin_counts: np.ndarray
    clip_bin_offsets: np.ndarray
    bin_failed_count: np.ndarray


class MotionSource:
    def __init__(
        self,
        *,
        dataset_paths: list[str],
        dataset_path_weights: list[float],
        env_size: int,
        sampling_mode: str = "adaptive",
        seed: int = 0,
        adaptive_kernel_size: int = 1,
        adaptive_lambda: float = 0.8,
        adaptive_uniform_ratio: float = 0.1,
        adaptive_alpha: float = 0.001,
    ) -> None:
        if sampling_mode not in {"start", "uniform", "adaptive"}:
            raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")
        if len(dataset_paths) == 0:
            raise ValueError("dataset_paths must be non-empty")
        if len(dataset_paths) != len(dataset_path_weights):
            raise ValueError("dataset_paths and dataset_path_weights must have the same length")

        self.datasets = [CompiledMotionDataset.open(path) for path in dataset_paths]
        joint_names = self.datasets[0].joint_names
        body_names = self.datasets[0].body_names
        fps = self.datasets[0].fps
        for dataset in self.datasets[1:]:
            if dataset.joint_names != joint_names:
                raise ValueError("All datasets must share joint_names")
            if dataset.body_names != body_names:
                raise ValueError("All datasets must share body_names")
            if dataset.fps != fps:
                raise ValueError("All datasets must share fps")

        self.joint_names = joint_names
        self.body_names = body_names
        self.fps = fps
        self.env_size = int(env_size)
        self.sampling_mode = sampling_mode
        self.adaptive_kernel_size = max(1, int(adaptive_kernel_size))
        self.adaptive_lambda = float(adaptive_lambda)
        self.adaptive_uniform_ratio = float(adaptive_uniform_ratio)
        self.adaptive_alpha = float(adaptive_alpha)
        self.rng = np.random.default_rng(seed)

        dataset_weights = np.asarray(dataset_path_weights, dtype=np.float64)
        if np.any(dataset_weights < 0) or np.isclose(dataset_weights.sum(), 0.0):
            raise ValueError("dataset_path_weights must be non-negative and sum to a positive value")
        self.dataset_probs = dataset_weights / dataset_weights.sum()

        self.active_dataset_ids = np.zeros((self.env_size,), dtype=np.int64)
        self.active_clip_ids = np.zeros((self.env_size,), dtype=np.int64)
        self.active_frame_idx = np.zeros((self.env_size,), dtype=np.int64)
        self.active_clip_num_frames = np.zeros((self.env_size,), dtype=np.int64)

        self._adaptive = [self._build_adaptive_state(dataset) for dataset in self.datasets]

    def _build_adaptive_state(self, dataset: CompiledMotionDataset) -> _DatasetAdaptiveState:
        clip_bin_counts = np.maximum(1, dataset.clip_num_frames // max(dataset.fps, 1) + 1).astype(np.int64)
        clip_bin_offsets = np.zeros((dataset.num_clips + 1,), dtype=np.int64)
        clip_bin_offsets[1:] = np.cumsum(clip_bin_counts)
        bin_failed_count = np.zeros((int(clip_bin_offsets[-1]),), dtype=np.float64)
        return _DatasetAdaptiveState(
            clip_bin_counts=clip_bin_counts,
            clip_bin_offsets=clip_bin_offsets,
            bin_failed_count=bin_failed_count,
        )

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        if env_ids is None:
            env_ids = np.arange(self.env_size, dtype=np.int64)
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if env_ids.size == 0:
            return

        dataset_ids = self.rng.choice(
            len(self.datasets),
            size=env_ids.shape[0],
            replace=True,
            p=self.dataset_probs,
        )
        self.active_dataset_ids[env_ids] = dataset_ids

        for dataset_id, dataset in enumerate(self.datasets):
            local_mask = dataset_ids == dataset_id
            if not np.any(local_mask):
                continue
            local_env_ids = env_ids[local_mask]
            clip_ids, frame_idx = self._sample_within_dataset(dataset_id, local_env_ids.shape[0])
            self.active_clip_ids[local_env_ids] = clip_ids
            self.active_frame_idx[local_env_ids] = frame_idx
            self.active_clip_num_frames[local_env_ids] = dataset.clip_num_frames[clip_ids]

    def _sample_within_dataset(self, dataset_id: int, count: int) -> tuple[np.ndarray, np.ndarray]:
        dataset = self.datasets[dataset_id]
        clip_probs = np.asarray(dataset.clip_weights, dtype=np.float64)
        clip_probs = clip_probs / clip_probs.sum()

        if self.sampling_mode == "start":
            clip_ids = self.rng.choice(dataset.num_clips, size=count, replace=True, p=clip_probs)
            return clip_ids.astype(np.int64), np.zeros((count,), dtype=np.int64)

        if self.sampling_mode == "uniform":
            clip_ids = self.rng.choice(dataset.num_clips, size=count, replace=True, p=clip_probs)
            starts = self._uniform_frame_start(dataset, clip_ids)
            return clip_ids.astype(np.int64), starts

        return self._adaptive_frame_start(dataset_id, count)

    def _uniform_frame_start(self, dataset: CompiledMotionDataset, clip_ids: np.ndarray) -> np.ndarray:
        max_start = np.clip(dataset.clip_num_frames[clip_ids] - 1, 0, None)
        starts = np.zeros_like(max_start, dtype=np.int64)
        non_zero = max_start > 0
        starts[non_zero] = self.rng.integers(0, max_start[non_zero] + 1, size=int(non_zero.sum()))
        return starts

    def _smoothed_bin_probabilities(self, dataset_id: int) -> np.ndarray:
        dataset = self.datasets[dataset_id]
        adaptive = self._adaptive[dataset_id]
        probs = np.zeros_like(adaptive.bin_failed_count, dtype=np.float64)
        kernel = np.asarray(
            [self.adaptive_lambda**idx for idx in range(self.adaptive_kernel_size)],
            dtype=np.float64,
        )
        kernel = kernel / kernel.sum()

        for clip_id in range(dataset.num_clips):
            start = adaptive.clip_bin_offsets[clip_id]
            end = adaptive.clip_bin_offsets[clip_id + 1]
            clip_probs = adaptive.bin_failed_count[start:end].copy()
            clip_probs += self.adaptive_uniform_ratio / max(adaptive.clip_bin_counts[clip_id], 1)
            clip_probs = np.clip(clip_probs, 0.0, None)
            if kernel.size > 1:
                clip_probs = np.convolve(clip_probs, kernel, mode="full")[: clip_probs.shape[0]]
            clip_probs *= float(dataset.clip_weights[clip_id])
            probs[start:end] = clip_probs

        total = probs.sum()
        if total <= 0.0:
            probs.fill(1.0 / probs.size)
        else:
            probs /= total
        return probs

    def _adaptive_frame_start(self, dataset_id: int, count: int) -> tuple[np.ndarray, np.ndarray]:
        dataset = self.datasets[dataset_id]
        adaptive = self._adaptive[dataset_id]
        probs = self._smoothed_bin_probabilities(dataset_id)
        sampled_flat_bins = self.rng.choice(probs.size, size=count, replace=True, p=probs)
        clip_ids = np.searchsorted(adaptive.clip_bin_offsets[1:], sampled_flat_bins, side="right").astype(np.int64)
        local_bin_ids = sampled_flat_bins - adaptive.clip_bin_offsets[clip_ids]

        frame_idx = np.zeros((count,), dtype=np.int64)
        for row, clip_id in enumerate(clip_ids):
            clip_len = int(dataset.clip_num_frames[clip_id])
            bin_count = int(adaptive.clip_bin_counts[clip_id])
            if clip_len <= 1 or bin_count <= 1:
                frame_idx[row] = 0
                continue
            max_start = max(clip_len - 1, 0)
            bin_start = int(np.floor(local_bin_ids[row] * max_start / bin_count))
            bin_end = int(np.floor((local_bin_ids[row] + 1) * max_start / bin_count))
            if bin_end <= bin_start:
                frame_idx[row] = bin_start
            else:
                frame_idx[row] = int(self.rng.integers(bin_start, bin_end + 1))
        return clip_ids, frame_idx

    def record_failures(self, env_ids: np.ndarray) -> None:
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if env_ids.size == 0:
            return

        for dataset_id in np.unique(self.active_dataset_ids[env_ids]):
            local_env_ids = env_ids[self.active_dataset_ids[env_ids] == dataset_id]
            adaptive = self._adaptive[int(dataset_id)]
            updates = np.zeros_like(adaptive.bin_failed_count, dtype=np.float64)
            for env_id in local_env_ids:
                clip_id = int(self.active_clip_ids[env_id])
                clip_len = max(int(self.active_clip_num_frames[env_id]), 1)
                bin_count = int(adaptive.clip_bin_counts[clip_id])
                local_bin = min(int(self.active_frame_idx[env_id] * bin_count // clip_len), bin_count - 1)
                flat_bin = int(adaptive.clip_bin_offsets[clip_id] + local_bin)
                updates[flat_bin] += 1.0
            adaptive.bin_failed_count *= 1.0 - self.adaptive_alpha
            adaptive.bin_failed_count += self.adaptive_alpha * updates

    def reference(
        self,
        env_ids: np.ndarray | None = None,
        *,
        frame_offsets: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        if env_ids is None:
            env_ids = np.arange(self.env_size, dtype=np.int64)
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if frame_offsets is None:
            frame_offsets = np.array([0], dtype=np.int64)
        frame_offsets = np.asarray(frame_offsets, dtype=np.int64).reshape(-1)

        merged: dict[str, np.ndarray] = {}
        for dataset_id, dataset in enumerate(self.datasets):
            local_mask = self.active_dataset_ids[env_ids] == dataset_id
            if not np.any(local_mask):
                continue
            local_env_ids = env_ids[local_mask]
            local_frames = self.active_frame_idx[local_env_ids]
            sl = dataset.get_frames(
                self.active_clip_ids[local_env_ids],
                local_frames,
                steps=frame_offsets,
            )
            for key, value in sl.items():
                if key not in merged:
                    merged[key] = np.zeros((env_ids.shape[0],) + value.shape[1:], dtype=value.dtype)
                merged[key][np.nonzero(local_mask)[0]] = value
        return merged

    def advance(self, env_ids: np.ndarray | None = None) -> None:
        if env_ids is None:
            env_ids = np.arange(self.env_size, dtype=np.int64)
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if env_ids.size == 0:
            return
        self.active_frame_idx[env_ids] = np.minimum(
            self.active_frame_idx[env_ids] + 1,
            np.clip(self.active_clip_num_frames[env_ids] - 1, 0, None),
        )

    def sampling_metrics(self) -> dict[str, np.ndarray]:
        if len(self.datasets) == 1:
            probs = self._smoothed_bin_probabilities(0)
        else:
            merged: list[np.ndarray] = []
            for dataset_id, dataset_prob in enumerate(self.dataset_probs):
                probs = self._smoothed_bin_probabilities(dataset_id) * float(dataset_prob)
                merged.append(probs)
            probs = np.concatenate(merged, axis=0)
            total = probs.sum()
            if total > 0.0:
                probs = probs / total
            else:
                probs.fill(1.0 / probs.size)

        entropy = -np.sum(probs * np.log(probs + 1e-12))
        entropy_norm = entropy / np.log(max(probs.size, 2))
        entropy_norm = float(np.clip(entropy_norm, 0.0, 1.0))
        top1_idx = int(np.argmax(probs))
        top1_prob = float(probs[top1_idx])
        return {
            "sampling_entropy": np.asarray(entropy_norm, dtype=np.float32),
            "sampling_top1_prob": np.asarray(top1_prob, dtype=np.float32),
            "sampling_top1_bin": np.asarray(top1_idx / max(probs.size, 1), dtype=np.float32),
        }
