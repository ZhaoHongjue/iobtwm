"""Tests for the inspection CLI, run against a small converted fake dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.preprocess import convert, inspect_unified
from tests.scripts.preprocess.test_pipeline import M3N, PARKLAND, _convert, _write_sources


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("inspect")
    src, out = base / "src", base / "out"
    _write_sources(src)
    _convert(src, out)
    convert.write_catalog(src, out, [])
    return out


def _run(root: Path, *args: str) -> int:
    return inspect_unified.main(["--root", str(root), *args])


def test_overview_lists_every_subset(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(root, "overview") == 0
    out = capsys.readouterr().out
    assert PARKLAND in out and M3N in out
    assert "4 subsets, 15 samples" in out


def test_labels_aggregates_per_dataset(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(root, "labels", "--dataset", "M3N-VC") == 0
    out = capsys.readouterr().out
    assert "== M3N-VC (3 samples)" in out and "cx30" in out


def test_subset_reports_metadata_and_extras(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(root, "subset", PARKLAND, "--stats-rows", "4") == 0
    out = capsys.readouterr().out
    assert "samples 6" in out
    assert "extras/wavelet/db4/shake/audio" in out


def test_sample_shows_original_and_writes_plot(
    root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plot = tmp_path / "s.png"
    sid = f"{M3N}/a06_cx30_run6_o1000_rs1_00001_9"
    assert _run(root, "sample", sid, "--original", "--plot", str(plot)) == 0
    out = capsys.readouterr().out
    assert "audio: Tensor (3200,) torch.int16" in out
    assert "offset_ms       1000" in out
    assert plot.stat().st_size > 0


def test_find_filters_by_label_and_node(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(root, "find", "--label", "kona", "--node", "rs1") == 0
    assert "3 matching samples" in capsys.readouterr().out


def test_unknown_subset_exits_with_hint(root: Path) -> None:
    with pytest.raises(SystemExit, match="no subset"):
        _run(root, "subset", "Nope/folder")
