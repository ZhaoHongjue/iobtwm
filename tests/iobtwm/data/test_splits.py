from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iobtwm.data.config import SplitConfig
from iobtwm.data.manifest import ClipTable, load_clip_table
from iobtwm.data.splits import (
    DROPPED,
    allocate_records,
    assign_splits,
    official_code,
    split_random_clip,
    split_time,
)
from tests.iobtwm.data.conftest import Row, write_subset


@pytest.fixture
def table(toy_root: Path) -> ClipTable:
    return load_clip_table(toy_root, ("Toy/plain", "Toy/named"))


def test_random_clip_counts_and_determinism() -> None:
    codes = split_random_clip(100, (0.8, 0.1, 0.1), seed=3)

    assert np.bincount(codes).tolist() == [80, 10, 10]
    assert np.array_equal(codes, split_random_clip(100, (0.8, 0.1, 0.1), seed=3))
    assert not np.array_equal(codes, split_random_clip(100, (0.8, 0.1, 0.1), seed=4))


def test_time_split_is_chronological_per_record(table: ClipTable) -> None:
    codes = split_time(table, (0.6, 0.2, 0.2), gap_seconds=0.0)

    run1 = table.record == table.records.index("Toy/plain/run1")
    order = np.argsort(table.clip[run1])
    assert codes[run1][order].tolist() == [0] * 6 + [1] * 2 + [2] * 2


def test_time_split_drops_clips_after_each_cut(table: ClipTable) -> None:
    codes = split_time(table, (0.6, 0.2, 0.2), gap_seconds=2.0)

    run1 = table.record == table.records.index("Toy/plain/run1")
    order = np.argsort(table.clip[run1])
    assert codes[run1][order].tolist() == [0] * 6 + [DROPPED, 1, DROPPED, 2]


def test_time_split_shares_cuts_across_nodes(table: ClipTable) -> None:
    codes = split_time(table, (0.5, 0.0, 0.5), gap_seconds=0.0)

    run0 = table.record == table.records.index("Toy/plain/run0")
    for node in np.unique(table.node[run0]):
        mask = run0 & (table.node == node)
        assert codes[mask][np.argsort(table.clip[mask])].tolist() == [0] * 5 + [2] * 5


def test_record_split_keeps_records_whole(table: ClipTable) -> None:
    codes = assign_splits(table, SplitConfig(strategy="record", ratios=(0.5, 0.0, 0.5)))

    for record in np.unique(table.record):
        assert len(np.unique(codes[table.record == record])) == 1
    for runs in (("run0", "run1"), ("run2", "run3")):
        mask = np.isin(table.record, [table.records.index(f"Toy/plain/{r}") for r in runs])
        assert sorted(np.unique(codes[mask]).tolist()) == [0, 2]


def test_record_split_sends_single_records_to_train(table: ClipTable) -> None:
    codes = assign_splits(table, SplitConfig(strategy="record"))

    assert (codes[table.record == table.records.index("Named/rec_x")] == 0).all()


@pytest.mark.parametrize(
    ("n", "ratios", "expected"),
    [
        (1, (0.8, 0.1, 0.1), (1, 0, 0)),
        (2, (0.8, 0.1, 0.1), (1, 0, 1)),
        (3, (0.8, 0.1, 0.1), (2, 0, 1)),
        (3, (0.4, 0.3, 0.3), (1, 1, 1)),
        (5, (0.6, 0.4, 0.0), (3, 2, 0)),
        (10, (0.8, 0.1, 0.1), (8, 1, 1)),
        (4, (0.5, 0.0, 0.5), (2, 0, 2)),
        (5, (1.0, 0.0, 0.0), (5, 0, 0)),
    ],
)
def test_allocate_records(
    n: int, ratios: tuple[float, float, float], expected: tuple[int, int, int]
) -> None:
    assert allocate_records(n, ratios) == expected


@pytest.mark.parametrize("ratios", [(0.5, 0.5, 0.5), (0.0, 0.5, 0.5), (-0.1, 0.6, 0.5)])
def test_split_config_rejects_bad_ratios(ratios: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError, match="ratio"):
        SplitConfig(ratios=ratios)


@pytest.mark.parametrize(
    ("dataset", "run", "expected"),
    [
        ("M3N-VC", 4, 0),
        ("M3N-VC", 5, 2),
        ("ICT", 1, 2),
        ("WestPoint", 8, 0),
        ("WestPoint", 9, 2),
        ("WestPoint", 3, 2),
        ("GracesQuarters", 0, None),
    ],
)
def test_official_code(dataset: str, run: int, expected: int | None) -> None:
    assert official_code(dataset, run) == expected


@pytest.fixture
def with_m3n(toy_root: Path) -> ClipTable:
    rows = [
        Row(f"{scene}_run{run}_n1_{clip}", ["cx30"], clip, run=run, scene=scene)
        for scene in ("x", "y")
        for run in range(4)
        for clip in range(3)
    ]
    write_subset(toy_root, "M3N-VC/xy", "M3N-VC", rows, audio_sr=20, seismic_sr=10)
    return load_clip_table(toy_root, ("Toy/plain", "Toy/named", "M3N-VC/xy"))


def _record_split(table: ClipTable, codes: np.ndarray, name: str) -> int:
    values = np.unique(codes[table.record == table.records.index(name)])
    assert len(values) == 1
    return int(values[0])


def test_official_split_uses_runs_and_falls_back_to_record(with_m3n: ClipTable) -> None:
    config = SplitConfig(strategy="official", ratios=(0.5, 0.0, 0.5))
    codes = assign_splits(with_m3n, config)

    for run in range(4):
        assert _record_split(with_m3n, codes, f"M3N-VC/xy/x/run{run}") == (0 if run % 2 == 0 else 2)
    toy = [_record_split(with_m3n, codes, f"Toy/plain/run{r}") for r in range(4)]
    assert sorted(toy) == [0, 0, 2, 2]


@pytest.mark.parametrize(
    ("holdout", "held"),
    [
        (("Named",), {"Named/rec_x"}),
        (("Toy/plain",), {f"Toy/plain/run{r}" for r in range(4)}),
        (("M3N-VC/x", "Named"), {"Named/rec_x"} | {f"M3N-VC/xy/x/run{r}" for r in range(4)}),
    ],
)
def test_holdout_split(with_m3n: ClipTable, holdout: tuple[str, ...], held: set[str]) -> None:
    config = SplitConfig(strategy="holdout", holdout=holdout, ratios=(0.8, 0.2, 0.0))
    codes = assign_splits(with_m3n, config)

    for name in held:
        assert _record_split(with_m3n, codes, name) == 2
    others = ~np.isin(with_m3n.record, [with_m3n.records.index(n) for n in held])
    assert set(np.unique(codes[others]).tolist()) <= {0, 1}
    assert 0 in codes[others]


def test_holdout_rejects_unknown_names(with_m3n: ClipTable) -> None:
    with pytest.raises(ValueError, match="match no recording"):
        assign_splits(with_m3n, SplitConfig(strategy="holdout", holdout=("Nowhere",)))


def test_holdout_rejects_holding_out_everything(toy_root: Path) -> None:
    table = load_clip_table(toy_root, ("Toy/plain",))

    with pytest.raises(ValueError, match="no recording for training"):
        assign_splits(table, SplitConfig(strategy="holdout", holdout=("Toy",)))


def test_holdout_needs_names() -> None:
    with pytest.raises(ValueError, match="holdout"):
        SplitConfig(strategy="holdout")
