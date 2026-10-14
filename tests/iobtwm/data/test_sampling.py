from __future__ import annotations

import numpy as np
import pytest

from iobtwm.data.config import SamplingConfig
from iobtwm.data.sampling import WindowGrid, WindowSampler, plan_windows, to_ticks, window_grid

CLIP_TICKS = 20  # 2 s at 10 ticks per second


@pytest.mark.parametrize(
    ("config", "train", "expected"),
    [
        (SamplingConfig(mode="clip"), True, WindowGrid(20, 20, False)),
        (SamplingConfig(mode="multi_clip", num_clips=4), True, WindowGrid(80, 80, False)),
        (
            SamplingConfig(mode="multi_clip", num_clips=4, clip_stride=1),
            False,
            WindowGrid(80, 20, False),
        ),
        (SamplingConfig(mode="random_crop", window_seconds=5.0), True, WindowGrid(50, 50, True)),
        (SamplingConfig(mode="random_crop", window_seconds=5.0), False, WindowGrid(50, 50, False)),
        (
            SamplingConfig(mode="random_crop", window_seconds=5.0, eval_stride_seconds=2.5),
            False,
            WindowGrid(50, 25, False),
        ),
    ],
)
def test_window_grid(config: SamplingConfig, train: bool, expected: WindowGrid) -> None:
    assert window_grid(config, CLIP_TICKS, 10, train=train) == expected


def test_to_ticks_rejects_fractions() -> None:
    with pytest.raises(ValueError, match="whole number"):
        to_ticks(0.05, 10, "window")


def test_plan_fixed_grid_skips_short_segments() -> None:
    lengths = np.array([60, 10, 45], dtype=np.int64)

    segment, start = plan_windows(lengths, WindowGrid(20, 20, False), rng=None)

    assert segment.tolist() == [0, 0, 0, 2, 2]
    assert start.tolist() == [0, 20, 40, 0, 20]


def test_plan_random_offset_stays_inside_without_overlap() -> None:
    lengths = np.array([100, 55, 50, 30], dtype=np.int64)
    grid = WindowGrid(50, 50, True)
    rng = np.random.default_rng(0)

    for _ in range(50):
        segment, start = plan_windows(lengths, grid, rng)
        assert (start >= 0).all() and (start + 50 <= lengths[segment]).all()
        assert sorted(set(segment.tolist())) == [0, 1, 2]
        assert (np.bincount(segment, minlength=4) <= lengths // 50).all()


def test_plan_random_offset_covers_every_start() -> None:
    lengths = np.array([200], dtype=np.int64)
    rng = np.random.default_rng(1)

    starts = {
        s
        for _ in range(300)
        for s in plan_windows(lengths, WindowGrid(50, 50, True), rng)[1].tolist()
    }

    assert starts == set(range(150 + 1))


def test_sampler_is_deterministic_per_epoch() -> None:
    def keys(epoch: int) -> list[tuple[int, int]]:
        sampler = WindowSampler(
            np.array([7, 9], dtype=np.int64),
            np.array([400, 300], dtype=np.int64),
            WindowGrid(50, 50, True),
            shuffle=True,
            seed=0,
        )
        sampler.set_epoch(epoch)
        return list(sampler)

    assert keys(1) == keys(1)
    assert keys(1) != keys(2)
    assert {segment for segment, _ in keys(1)} == {7, 9}


def test_unshuffled_sampler_keeps_order() -> None:
    sampler = WindowSampler(
        np.array([3, 5], dtype=np.int64),
        np.array([40, 20], dtype=np.int64),
        WindowGrid(20, 20, False),
        shuffle=False,
        seed=0,
    )

    assert list(sampler) == [(3, 0), (3, 20), (5, 0)]
    assert len(sampler) == 3
