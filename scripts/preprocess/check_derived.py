"""Check that the excluded frequency-domain folders are derivable from converted data.

The folders in ``specs.DERIVED_FROM`` were not converted because each of their samples is the
per-interval FFT of a time-domain sample with the same filename. This script proves that for
every file: same label and flags, and spectra equal to the FFT of the *unified* time-domain
signal up to float32 rounding (the source FFTs were computed before casting to float32, so an
exact match is not possible).

Usage:
    uv run python -m scripts.preprocess.check_derived --src datasets --out datasets_unified
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import h5py
import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import torch

from scripts.preprocess import codec
from scripts.preprocess.convert import map_ordered, read_source
from scripts.preprocess.layout import MANIFEST_FILE, SIGNALS_FILE, list_samples, write_json
from scripts.preprocess.specs import DERIVED_FROM

log = logging.getLogger(__name__)

# Max |FFT(time) - stored spectrum| relative to the spectrum's max magnitude.
REL_TOL: Final = 1e-4
FILES_PER_TASK: Final = 256


@dataclass(frozen=True, slots=True)
class Task:
    """A block of frequency files and the unified metadata of their time-domain twins."""

    src: str
    out: str
    freq_folder: str
    time_folder: str
    items: tuple[tuple[str, str], ...]  # (filename, original_meta of the time sample)


@dataclass(frozen=True, slots=True)
class Result:
    """Outcome of one :class:`Task`."""

    checked: int
    max_rel_err: float
    failures: tuple[str, ...]


def _spectrum(time: npt.NDArray[np.generic]) -> npt.NDArray[np.float64]:
    """[1, i, s] waveform intervals -> [2, i, s] (real, imag) FFT, as the source did."""
    f = np.fft.fft(time.astype(np.float64)[0], axis=-1)
    return np.stack([f.real, f.imag])


def _check_one(
    freq: dict[str, object], time: dict[str, object], name: str
) -> tuple[float, list[str]]:
    failures = [f"{name}: {d}" for k in ("label", "flag") for d in codec.diff(freq[k], time[k])]
    worst = 0.0
    fdata, tdata = freq["data"], time["data"]
    assert isinstance(fdata, dict) and isinstance(tdata, dict)
    for node, mods in fdata.items():
        assert isinstance(mods, dict)
        for mod, spec in mods.items():
            assert isinstance(spec, torch.Tensor)
            t = tdata[node][mod]
            assert isinstance(t, torch.Tensor)
            # dwt variants store [1, 1, 16000]; the FFT was taken per 0.2 s interval.
            intervals = t.numpy().reshape(1, spec.shape[1], spec.shape[2])
            expected = _spectrum(intervals)
            scale = max(float(np.abs(expected).max()), 1e-12)
            err = float(np.abs(spec.numpy().astype(np.float64) - expected).max()) / scale
            worst = max(worst, err)
            if err > REL_TOL:
                failures.append(f"{name}: {mod} spectrum off by {err:.2e} (relative)")
    return worst, failures


def check_task(task: Task) -> Result:
    """Check one block (runs in worker processes)."""
    worst = 0.0
    failures: list[str] = []
    with h5py.File(Path(task.out) / task.time_folder / SIGNALS_FILE, "r") as h5:

        def fetch(key: str, row: int) -> npt.NDArray[np.generic]:
            arr: npt.NDArray[np.generic] = h5[key][row]
            return arr

        for filename, meta in task.items:
            freq, _, _ = read_source(Path(task.src) / task.freq_folder / filename)
            time = codec.decode(json.loads(meta), fetch)
            assert isinstance(freq, dict) and isinstance(time, dict)
            err, fails = _check_one(freq, time, f"{task.freq_folder}/{filename}")
            worst, failures = max(worst, err), failures + fails
    return Result(len(task.items), worst, tuple(failures))


def check_folder(
    src: Path, out: Path, freq_folder: str, time_folder: str, workers: int
) -> dict[str, object]:
    """Check every file of one derived folder against its time-domain subset."""
    manifest = pq.read_table(
        out / time_folder / MANIFEST_FILE, columns=["source_file", "original_meta"]
    )
    meta_by_name = {
        Path(str(f)).name: str(m)
        for f, m in zip(
            manifest["source_file"].to_pylist(), manifest["original_meta"].to_pylist(), strict=True
        )
    }
    names = [p.name for p in list_samples(src / freq_folder)]
    failures = [
        f"{freq_folder}/{n}: no time-domain twin in {time_folder}"
        for n in names
        if n not in meta_by_name
    ]
    items = [(n, meta_by_name[n]) for n in names if n in meta_by_name]
    tasks = [
        Task(str(src), str(out), freq_folder, time_folder, tuple(items[i : i + FILES_PER_TASK]))
        for i in range(0, len(items), FILES_PER_TASK)
    ]
    worst, checked = 0.0, 0
    for res in map_ordered(check_task, tasks, workers):
        worst, checked = max(worst, res.max_rel_err), checked + res.checked
        failures += res.failures
    log.info(
        "%s: %d files, max relative error %.2e, %d failures",
        freq_folder,
        checked,
        worst,
        len(failures),
    )
    return {
        "derived_from": time_folder,
        "files": len(names),
        "checked": checked,
        "max_rel_err": worst,
        "failures": failures[:100],
        "num_failures": len(failures),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src", type=Path, default=Path("datasets"))
    p.add_argument("--out", type=Path, default=Path("datasets_unified"))
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    results = {
        f: check_folder(args.src, args.out, f, t, args.workers) for f, t in DERIVED_FROM.items()
    }
    ok = all(r["num_failures"] == 0 for r in results.values())
    write_json({"ok": ok, "rel_tol": REL_TOL, "folders": results}, args.out / "derived_report.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
