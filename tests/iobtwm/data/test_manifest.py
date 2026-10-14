from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iobtwm.data.manifest import load_clip_table, normalize_label, record_key
from tests.iobtwm.data.conftest import Row, write_subset

SUBSETS = ("Toy/plain", "Toy/named")


def test_load_drops_wavelet_and_offset_rows(toy_root: Path) -> None:
    table = load_clip_table(toy_root, SUBSETS)

    assert len(table) == 42 + 4
    named = table.subset == 1
    assert table.row[named].tolist() == [0, 3, 6, 9]


def test_load_builds_vocabularies(toy_root: Path) -> None:
    table = load_clip_table(toy_root, SUBSETS)

    assert table.classes == ("a", "b", "warthog")
    assert table.datasets == ("Named", "Toy")
    assert "Named/rec_x" in table.records
    assert "Toy/plain/run2" in table.records
    assert table.clip_seconds == 2.0


def test_load_reads_distance_and_clip_start(toy_root: Path) -> None:
    table = load_clip_table(toy_root, SUBSETS)

    plain = table.subset == 0
    assert table.distance[plain][:3].tolist() == [0.0, 1.0, 2.0]
    assert np.isnan(table.distance[~plain]).all()
    assert table.clip_start[~plain].tolist() == [100.0, 102.0, 104.0, 106.0]
    assert np.isnan(table.clip_start[plain]).all()


def test_load_rejects_duplicate_clips(tmp_path: Path) -> None:
    rows = [Row("x_0", ["a"], 0, run=0), Row("x_0_again", ["a"], 0, run=0)]
    write_subset(tmp_path, "Dup/x", "Dup", rows, audio_sr=20, seismic_sr=10)

    with pytest.raises(ValueError, match="twice"):
        load_clip_table(tmp_path, ("Dup/x",))


def test_load_rejects_mixed_clip_durations(tmp_path: Path) -> None:
    write_subset(tmp_path, "A/x", "A", [Row("a", ["a"], 0, run=0)], audio_sr=20, seismic_sr=10)
    write_subset(tmp_path, "B/x", "B", [Row("b", ["a"], 0, run=0)], audio_sr=20, seismic_sr=10)
    info = tmp_path / "B/x/info.json"
    info.write_text(info.read_text().replace("[40]", "[60]").replace("[20]", "[30]"))

    with pytest.raises(ValueError, match="clip durations"):
        load_clip_table(tmp_path, ("A/x", "B/x"))


@pytest.mark.parametrize(
    ("label", "expected"), [("Warhog", "warthog"), ("polairs", "polaris"), ("CX30", "cx30")]
)
def test_normalize_label(label: str, expected: str) -> None:
    assert normalize_label(label) == expected


def test_record_key_prefers_recording_name() -> None:
    assert record_key("P/individual", "P", "motor", None) == "P/motor"
    assert record_key("P/extra", "P", "motor", None) == "P/motor"
    assert record_key("G/day_1", "G", None, 3) == "G/day_1/run3"
    assert record_key("M/xy", "M", None, 3, "x") == "M/xy/x/run3"


def test_record_key_requires_an_identifier() -> None:
    with pytest.raises(ValueError, match="neither"):
        record_key("G/day_1", "G", None, None)
