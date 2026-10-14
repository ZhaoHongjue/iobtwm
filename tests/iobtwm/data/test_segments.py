from __future__ import annotations

from pathlib import Path

import numpy as np

from iobtwm.data.manifest import load_clip_table
from iobtwm.data.segments import build_segments
from iobtwm.data.splits import DROPPED


def _segment_clips(toy_root: Path, codes_fn: str = "train") -> list[tuple[str, list[int]]]:
    table = load_clip_table(toy_root, ("Toy/plain",))
    codes = np.zeros(len(table), dtype=np.int8)
    if codes_fn == "split_run1":
        run1 = table.record == table.records.index("Toy/plain/run1")
        codes[run1 & (table.clip >= 5)] = 2
        codes[run1 & (table.clip == 8)] = DROPPED
    segments = build_segments(table, codes)
    out = []
    for s in range(len(segments)):
        clips = segments.clips[segments.offsets[s] : segments.offsets[s + 1]]
        name = f"{table.records[table.record[clips[0]]]}/{table.nodes[table.node[clips[0]]]}"
        out.append((name, table.clip[clips].tolist()))
    return out


def test_segments_follow_record_node_and_gaps(toy_root: Path) -> None:
    assert _segment_clips(toy_root) == [
        ("Toy/plain/run0/n1", list(range(10))),
        ("Toy/plain/run0/n2", list(range(10))),
        ("Toy/plain/run1/n1", list(range(10))),
        ("Toy/plain/run2/n1", [0, 1, 2]),
        ("Toy/plain/run2/n1", [4, 5, 6]),
        ("Toy/plain/run3/n1", list(range(6))),
    ]


def test_segments_break_at_split_changes_and_drop_clips(toy_root: Path) -> None:
    run1 = [
        c for name, c in _segment_clips(toy_root, "split_run1") if name.startswith("Toy/plain/run1")
    ]

    assert run1 == [[0, 1, 2, 3, 4], [5, 6, 7], [9]]


def test_segments_break_when_clip_starts_jump(toy_root: Path) -> None:
    table = load_clip_table(toy_root, ("Toy/named",))
    table.clip_start[2:] += 5.0

    segments = build_segments(table, np.zeros(len(table), dtype=np.int8))

    assert segments.num_clips().tolist() == [2, 2]


def test_segments_of_nothing(toy_root: Path) -> None:
    table = load_clip_table(toy_root, ("Toy/plain",))

    segments = build_segments(table, np.full(len(table), DROPPED, dtype=np.int8))

    assert len(segments) == 0
    assert segments.num_clips().tolist() == []
