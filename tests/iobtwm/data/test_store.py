from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iobtwm.data.manifest import load_clip_table
from iobtwm.data.segments import build_segments
from iobtwm.data.store import clip_scales, load_store, open_store, resample, save_store


def test_resample_identity_and_length() -> None:
    wave = np.arange(80, dtype=np.float64)

    assert np.array_equal(resample(wave, 20.0, 20), wave.astype(np.float32))
    assert len(resample(wave, 40.0, 20)) == 40
    assert len(resample(wave, 20.0, 50)) == 200


def test_resample_keeps_a_low_tone() -> None:
    t = np.arange(8000) / 8000
    tone = np.sin(2 * np.pi * 50 * t)

    out = resample(tone, 8000.0, 1600)

    expected = np.sin(2 * np.pi * 50 * np.arange(1600) / 1600)
    assert np.abs(out[100:-100] - expected[100:-100]).max() < 1e-2


def test_resample_rejects_fractional_rates() -> None:
    with pytest.raises(ValueError, match="integer"):
        resample(np.zeros(4), 100.5, 100)


def test_load_store_joins_rows_in_time_order(toy_root: Path) -> None:
    table = load_clip_table(toy_root, ("Toy/plain",))
    segments = build_segments(table, np.zeros(len(table), dtype=np.int8))

    store = load_store(toy_root, table, segments, {"audio": 20, "seismic": 10})

    first = store.segment("audio", 0)
    rows = table.row[segments.clips[: segments.offsets[1]]]
    expected = (rows[:, None] * 1000 + np.arange(40)).reshape(-1)
    assert np.array_equal(first, expected.astype(np.float32))
    assert len(store.segment("seismic", 0)) == 10 * 20


def test_load_store_resamples_whole_segments(toy_root: Path) -> None:
    table = load_clip_table(toy_root, ("Toy/named",))
    segments = build_segments(table, np.zeros(len(table), dtype=np.int8))

    store = load_store(toy_root, table, segments, {"audio": 20, "seismic": 10})

    assert store.offsets["audio"].tolist() == [0, 4 * 40]


def test_clip_scales_uses_train_medians_with_fallback() -> None:
    rms = np.array([1.0, 3.0, 2.0, 100.0, 0.0, 5.0, 7.0])
    keys = np.array([0, 0, 0, 0, 1, 1, 2])
    train = np.array([True, True, True, False, True, False, True])

    scales = clip_scales(rms, keys, train)

    # Group 0: median of train clips 1, 3, 2. Group 1 has only a silent train clip, so it gets
    # the median of the other groups (2 and 7). Group 2: its one train clip.
    assert scales.tolist() == [2.0, 2.0, 2.0, 2.0, 4.5, 4.5, 7.0]


def test_save_and_open_round_trip(toy_root: Path, tmp_path: Path) -> None:
    table = load_clip_table(toy_root, ("Toy/plain",))
    segments = build_segments(table, np.zeros(len(table), dtype=np.int8))
    store = load_store(toy_root, table, segments, {"audio": 20, "seismic": 10})

    save_store(store, tmp_path / "cache")
    reopened = open_store(tmp_path / "cache")

    assert reopened.rates == store.rates
    for modality in store.rates:
        assert np.array_equal(reopened.signals[modality], store.signals[modality])
        assert np.array_equal(reopened.offsets[modality], store.offsets[modality])
    assert not (tmp_path / "cache.tmp").exists()
