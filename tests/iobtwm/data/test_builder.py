from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from iobtwm.data import builder
from iobtwm.data.builder import build_data
from iobtwm.data.config import (
    DataConfig,
    SamplingConfig,
    SamplingMode,
    SpectralConfig,
    SpectralPlacement,
    SplitConfig,
    SplitStrategy,
)

SAMPLING = {
    "clip": SamplingConfig(mode="clip"),
    "multi_clip": SamplingConfig(mode="multi_clip", num_clips=2),
    "random_crop": SamplingConfig(mode="random_crop", window_seconds=3.0),
}


def _config(root: Path, **changes: object) -> DataConfig:
    base = DataConfig(
        root=root,
        subsets=("Toy/plain", "Toy/named"),
        audio_rate=20,
        seismic_rate=10,
        split=SplitConfig(strategy="time", ratios=(0.6, 0.2, 0.2), gap_seconds=0.0),
        sampling=SAMPLING["clip"],
        batch_size=4,
        num_workers=0,
    )
    return dataclasses.replace(base, **changes)  # type: ignore[arg-type]  # test helper


@pytest.mark.parametrize("strategy", ["random_clip", "time", "record"])
@pytest.mark.parametrize("mode", ["clip", "multi_clip", "random_crop"])
@pytest.mark.parametrize("placement", ["loader", "device"])
def test_every_option_yields_batches(
    toy_root: Path, strategy: SplitStrategy, mode: SamplingMode, placement: SpectralPlacement
) -> None:
    config = _config(
        toy_root,
        split=SplitConfig(strategy=strategy, ratios=(0.6, 0.2, 0.2), gap_seconds=0.0),
        sampling=SAMPLING[mode],
        spectral=SpectralConfig(placement=placement),
    )
    data = build_data(config)

    batch = next(data.iterate("train", epoch=1))

    window = {"clip": 40, "multi_clip": 80, "random_crop": 60}[mode]
    assert batch["audio"].shape[1:] == data.transform.output_shape("audio", window)
    assert batch["seismic"].shape[1:] == data.transform.output_shape("seismic", window // 2)
    assert batch["labels"].shape[1:] == (len(data.classes),)
    assert batch["distance"].dtype == torch.float32


def test_clip_mode_returns_the_original_clips(toy_root: Path) -> None:
    config = _config(
        toy_root,
        subsets=("Toy/plain",),
        normalization="none",
        remove_dc=False,
        split=SplitConfig(strategy="record", ratios=(1.0, 0.0, 0.0)),
        sampling=SamplingConfig(mode="clip"),
        spectral=SpectralConfig(representation="waveform"),
    )
    data = build_data(config)

    seen = set()
    for batch in data.iterate("train"):
        rows = (batch["audio"][:, 0] / 1000).round().to(torch.int64)
        expected = rows[:, None] * 1000 + torch.arange(40)
        assert torch.equal(batch["audio"], expected.to(torch.float32))
        assert torch.equal(batch["seismic"], -(rows[:, None] * 1000 + torch.arange(20)).float())
        seen.update(rows.tolist())
    assert seen == set(range(42))


def test_labels_and_distance_cover_the_window(toy_root: Path) -> None:
    config = _config(
        toy_root,
        subsets=("Toy/plain",),
        split=SplitConfig(strategy="record", ratios=(1.0, 0.0, 0.0)),
        sampling=SamplingConfig(mode="multi_clip", num_clips=3),
    )
    data = build_data(config)
    dataset = data.loaders["train"].dataset
    segment = int(np.flatnonzero(data.segments.num_clips() == 6)[0])  # run3

    example = dataset[(segment, 20)]  # clips 1-3 of run3, distances 31-33

    assert example["labels"].tolist() == [0.0, 1.0]
    assert example["distance"].item() == 31.0
    assert example["start_seconds"].item() == 2.0


def test_device_placement_matches_loader_placement(toy_root: Path) -> None:
    on_loader = build_data(_config(toy_root, spectral=SpectralConfig(placement="loader")))
    on_device = build_data(_config(toy_root, spectral=SpectralConfig(placement="device")))

    for a, b in zip(on_loader.iterate("val"), on_device.iterate("val"), strict=True):
        torch.testing.assert_close(a["audio"], b["audio"])
        torch.testing.assert_close(a["seismic"], b["seismic"])


def test_random_crop_moves_between_epochs(toy_root: Path) -> None:
    data = build_data(_config(toy_root, sampling=SAMPLING["random_crop"]))

    def starts(epoch: int) -> list[float]:
        return sorted(
            t for b in data.iterate("train", epoch=epoch) for t in b["start_seconds"].tolist()
        )

    assert starts(1) == starts(1)
    assert starts(1) != starts(2)


def test_cache_is_reused(toy_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(toy_root, cache_dir=tmp_path / "cache")
    first = build_data(config)

    def fail(*_: object) -> None:
        raise AssertionError("signals were loaded again instead of using the cache")

    monkeypatch.setattr(builder, "load_store", fail)
    second = build_data(config)

    for modality in ("audio", "seismic"):
        assert np.array_equal(first.store.signals[modality], second.store.signals[modality])


def test_random_clip_split_leaves_no_long_windows(
    toy_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = _config(
        toy_root,
        split=SplitConfig(strategy="random_clip", ratios=(0.5, 0.25, 0.25)),
        sampling=SamplingConfig(mode="multi_clip", num_clips=6),
    )

    data = build_data(config)

    assert "test" not in data.loaders
    assert "no segment is long enough" in caplog.text
    assert data.summary()["test"].examples == 0
