"""Configuration for the training data pipeline.

Every choice that matters for experiments is a field here, so it can be swept later without
touching code: which subsets to use, how to split them, how to cut training windows, and how to
turn waveforms into model inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

Modality = Literal["audio", "seismic"]
SplitName = Literal["train", "val", "test"]
SplitStrategy = Literal["random_clip", "time", "record", "official", "holdout"]
StratifyKey = Literal["dataset", "scene", "labels"]
SamplingMode = Literal["clip", "multi_clip", "random_crop"]
Representation = Literal["waveform", "magnitude", "log_magnitude", "complex"]
WindowFunction = Literal["hann", "rect"]
SpectralPlacement = Literal["loader", "device"]
Normalization = Literal["none", "node", "dataset"]

SPLIT_NAMES: Final[tuple[SplitName, ...]] = ("train", "val", "test")

# Waveform subsets without duplicated recordings (see docs/tasks). Excluded on purpose:
# GQ *Filtered*/random/Mixed (subsets of sensor_all or synthetic), 2024-08-07-GQ_past (contained
# in 2024-08-07-GQ), ICT_*_dis (same session as M3N i29), ict_pt_files (synthetic), Parkland
# individual_samples (FFT only) and individual_time_samples_with_wind (synthetic wind).
DEFAULT_SUBSETS: Final[tuple[str, ...]] = (
    "Parkland/individual_time_samples",
    "Parkland/extra_time_samples",
    "GracesQuarters/day_1_sensor_all",
    "GracesQuarters/day_2_sensor_all",
    "GracesQuarters/2024-07-19-GQ-Background/individual_time_samples",
    "GracesQuarters/2024-08-05-GQ/individual_time_samples",
    "GracesQuarters/2024-08-06-GQ/individual_time_samples",
    "GracesQuarters/2024-08-07-GQ/individual_time_samples",
    "2026-04-14-West-Point/individual_time_samples",
    "M3N-VC/a06/individual_time_samples",
    "M3N-VC/h08/individual_time_samples",
    "M3N-VC/h24/individual_time_samples",
    "M3N-VC/i22/individual_time_samples",
    "M3N-VC/i29/individual_time_samples",
    "M3N-VC/s31/individual_time_samples",
)


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """How clips are assigned to train/val/test.

    Attributes:
        strategy: ``random_clip`` assigns every clip independently at random, as the early group
            papers did on MOD (leaks neighbouring clips across splits). ``time`` cuts each
            recording chronologically into train, then val, then test. ``record`` keeps every
            recording (run) whole and assigns recordings to splits. ``official`` uses the
            published run-level splits (M3N-VC, ICT, West Point: their held-out runs become
            test) and falls back to ``record`` for the other datasets. ``holdout`` sends every
            recording matching ``holdout`` to test and splits the rest into train and val by
            recording, to measure generalisation to an unseen site.
        ratios: Fractions for (train, val, test). A zero ratio yields an empty split. With
            ``holdout`` only the train:val proportion is used.
        seed: Seed for the random assignment.
        gap_seconds: ``time`` only. Clips this close after a cut are dropped, so neighbouring
            audio does not end up on both sides of a boundary.
        stratify: ``record`` only. Recordings are shuffled and divided within each group that
            shares these keys. A group with a single recording goes entirely to train. Adding
            ``scene`` keeps every M3N-VC scene in both train and test, but most groups then hold
            two recordings and split 1:1, far from ``ratios``.
        holdout: ``holdout`` only. Names of what to hold out: a dataset (``GracesQuarters``), a
            subset path (``GracesQuarters/day_2_sensor_all``) or a dataset scene
            (``M3N-VC/s31``).
    """

    strategy: SplitStrategy = "time"
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 0
    gap_seconds: float = 10.0
    stratify: tuple[StratifyKey, ...] = ("dataset", "labels")
    holdout: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(r < 0 for r in self.ratios) or not math.isclose(sum(self.ratios), 1.0):
            raise ValueError(f"split ratios must be non-negative and sum to 1, got {self.ratios}")
        if self.ratios[0] == 0:
            raise ValueError("the train ratio must be positive")
        if self.strategy == "holdout" and not self.holdout:
            raise ValueError("the holdout strategy needs at least one name in holdout")
        if self.gap_seconds < 0:
            raise ValueError(f"gap_seconds must be non-negative, got {self.gap_seconds}")


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """How training examples are cut out of the continuous recordings.

    Attributes:
        mode: ``clip`` yields single original clips (2 s). ``multi_clip`` yields ``num_clips``
            consecutive clips concatenated. ``random_crop`` yields windows of ``window_seconds``
            starting anywhere; during training each epoch uses a grid with a fresh random offset
            per recording, so every moment is used about once per epoch.
        num_clips: ``multi_clip`` only. Clips per example.
        clip_stride: ``multi_clip`` only. Clips between consecutive example starts; defaults to
            ``num_clips`` (non-overlapping, as FOCAL's sequence sampler).
        window_seconds: ``random_crop`` only. Example length.
        eval_stride_seconds: ``random_crop`` only. Stride of the fixed evaluation grid; defaults
            to ``window_seconds``.
    """

    mode: SamplingMode = "multi_clip"
    num_clips: int = 8
    clip_stride: int | None = None
    window_seconds: float = 24.0
    eval_stride_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.num_clips < 1:
            raise ValueError(f"num_clips must be at least 1, got {self.num_clips}")
        if self.clip_stride is not None and self.clip_stride < 1:
            raise ValueError(f"clip_stride must be at least 1, got {self.clip_stride}")
        if self.window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {self.window_seconds}")
        if self.eval_stride_seconds is not None and self.eval_stride_seconds <= 0:
            raise ValueError(
                f"eval_stride_seconds must be positive, got {self.eval_stride_seconds}"
            )


@dataclass(frozen=True, slots=True)
class SpectralConfig:
    """How waveforms become model inputs.

    The STFT runs at run time. With ``placement="loader"`` it runs in the DataLoader workers and
    batches arrive as spectra; with ``placement="device"`` batches arrive as waveforms and
    ``IoBTData.prepare`` computes the spectra on the target device (usually the GPU).

    Attributes:
        representation: ``waveform`` (no transform), ``magnitude`` (|X|), ``log_magnitude``
            (log(|X| + eps)) or ``complex`` (real and imaginary parts as two channels).
        window_seconds: STFT frame length.
        hop_seconds: STFT frame step.
        window: ``hann`` or ``rect``. ``rect`` with window = hop = 0.2 s reproduces the
            per-interval FFT of FOCAL / cs537aiot.
        log_eps: Added before the log in ``log_magnitude``.
        placement: Where the STFT runs, see above.
    """

    representation: Representation = "log_magnitude"
    window_seconds: float = 0.2
    hop_seconds: float = 0.1
    window: WindowFunction = "hann"
    log_eps: float = 1e-6
    placement: SpectralPlacement = "device"

    def __post_init__(self) -> None:
        if self.window_seconds <= 0 or self.hop_seconds <= 0:
            raise ValueError("STFT window_seconds and hop_seconds must be positive")
        if self.log_eps <= 0:
            raise ValueError(f"log_eps must be positive, got {self.log_eps}")


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Everything needed to build the train/val/test data loaders.

    Attributes:
        root: The ``datasets_unified`` directory.
        subsets: Unified subsets to load, as paths relative to ``root``.
        modalities: Modalities returned in each example.
        audio_rate: Common audio sample rate in Hz; every subset is resampled to it.
        seismic_rate: Common seismic sample rate in Hz.
        normalization: Per-channel scale estimated on the train split: one scale per
            (dataset, node), one per dataset, or none.
        remove_dc: Subtract each example's mean per modality.
        split: Split settings.
        sampling: Example cutting settings.
        spectral: Model input settings.
        batch_size: Batch size of every loader.
        num_workers: DataLoader worker processes.
        cache_dir: If set, the resampled and normalised signals are cached here and memory-mapped
            on later runs with the same settings.
    """

    root: Path = Path("datasets_unified")
    subsets: tuple[str, ...] = DEFAULT_SUBSETS
    modalities: tuple[Modality, ...] = ("audio", "seismic")
    audio_rate: int = 1600
    seismic_rate: int = 100
    normalization: Normalization = "node"
    remove_dc: bool = True
    split: SplitConfig = field(default_factory=SplitConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    batch_size: int = 64
    num_workers: int = 8
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if not self.subsets:
            raise ValueError("at least one subset is required")
        if not self.modalities or len(set(self.modalities)) != len(self.modalities):
            raise ValueError(f"modalities must be non-empty and unique, got {self.modalities}")
        if self.audio_rate <= 0 or self.seismic_rate <= 0:
            raise ValueError("sample rates must be positive")
        if self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("batch_size must be >= 1 and num_workers >= 0")

    def rate(self, modality: Modality) -> int:
        """Return the common sample rate of ``modality`` in Hz."""
        return self.audio_rate if modality == "audio" else self.seismic_rate
