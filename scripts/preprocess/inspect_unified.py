"""Inspect the unified dataset produced by ``convert``.

Subcommands:
    overview                    one line per subset: size, shapes, dtypes, rates, top labels
    labels [--dataset D]        label counts per dataset, and which subsets they come from
    subset FOLDER               metadata and signal statistics for one subset
    sample SAMPLE_ID [--plot P] one sample: metadata, signal stats, original structure, plot
    find [--label L] [...]      search samples by label / node / run / dataset

Usage:
    uv run python -m scripts.preprocess.inspect_unified overview
    uv run python -m scripts.preprocess.inspect_unified subset M3N-VC/a06/individual_time_samples
    uv run python -m scripts.preprocess.inspect_unified sample \\
        M3N-VC/a06/individual_time_samples/a06_cx30_run6_rs1_00000_9 --plot sample.png
    uv run python -m scripts.preprocess.inspect_unified find --label cx30 --node rs1
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

import h5py
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from scripts.preprocess import codec
from scripts.preprocess.layout import CATALOG_FILE, INFO_FILE, MANIFEST_FILE, SIGNALS_FILE

STATS_ROWS: Final = 1000
TOP_LABELS: Final = 4
MODALITIES: Final = ("audio", "seismic")

# Reference palette (dataviz skill), light mode.
SURFACE: Final = "#fcfcfb"
TEXT_PRIMARY: Final = "#0b0b0b"
TEXT_SECONDARY: Final = "#52514e"
GRID: Final = "#e4e3df"
SERIES: Final = {"audio": "#2a78d6", "seismic": "#eb6834"}
SEQUENTIAL_CMAP: Final = "Blues"


# ---------------------------------------------------------------------------- reading


@dataclass(frozen=True, slots=True)
class Subset:
    """A converted subset on disk."""

    root: Path
    folder: str

    @property
    def path(self) -> Path:
        """Directory holding the subset's files."""
        return self.root / self.folder

    def info(self) -> dict[str, object]:
        """Parsed ``info.json``."""
        info: dict[str, object] = json.loads((self.path / INFO_FILE).read_text())
        return info

    def manifest(self, columns: Sequence[str] | None = None) -> pa.Table:
        """The manifest table (optionally only some columns)."""
        return pq.read_table(
            self.path / MANIFEST_FILE, columns=None if columns is None else list(columns)
        )


def subsets(root: Path) -> list[Subset]:
    """All converted subsets listed in the catalog."""
    catalog = json.loads((root / CATALOG_FILE).read_text())
    return [Subset(root, f) for f in catalog["subsets"]]


def find_subset(root: Path, folder: str) -> Subset:
    """The subset for ``folder``.

    Raises:
        SystemExit: With a hint, if no such subset exists.
    """
    folder = folder.strip("/")
    known = {s.folder: s for s in subsets(root)}
    if folder not in known:
        close = [f for f in known if folder.lower() in f.lower()]
        raise SystemExit(
            f"no subset {folder!r}; did you mean: {', '.join(close) or 'see overview'}"
        )
    return known[folder]


def _fmt_shape(col: object) -> str:
    assert isinstance(col, dict)
    shape, dtype = col["row_shape"], np.dtype(str(col["dtype"]))
    assert isinstance(shape, list)
    return f"{'x'.join(map(str, shape))} {dtype.name}"


def _num(v: object) -> float:
    assert isinstance(v, int | float)
    return float(v)


# ---------------------------------------------------------------------------- overview / labels


def cmd_overview(root: Path) -> None:
    """Print one line per subset, then totals and catalog-level facts."""
    print(f"{'subset':<66} {'N':>7} {'audio':>18} {'seismic':>14} {'sr (Hz)':>11}  top labels")
    total = 0
    for s in subsets(root):
        info = s.info()
        cols = info["columns"]
        assert isinstance(cols, dict)
        counts = info["label_counts"]
        assert isinstance(counts, dict)
        top = ", ".join(f"{k} {v}" for k, v in list(counts.items())[:TOP_LABELS])
        more = f" (+{len(counts) - TOP_LABELS})" if len(counts) > TOP_LABELS else ""
        rates = f"{_num(info['audio_sr']):g}/{_num(info['seismic_sr']):g}"
        n = int(_num(info["num_samples"]))
        total += n
        rep = "" if info["representation"] == "waveform" else " [fft]"
        print(
            f"{s.folder + rep:<66} {n:>7} {_fmt_shape(cols['signals/audio']):>18} "
            f"{_fmt_shape(cols['signals/seismic']):>14} {rates:>11}  {top}{more}"
        )
    catalog = json.loads((root / CATALOG_FILE).read_text())
    print(f"\n{len(catalog['subsets'])} subsets, {total} samples")
    print("excluded:", *(f"\n  {k}: {v}" for k, v in catalog["excluded"].items()))
    print("not converted files:", catalog["not_converted_files"])


def _short(folder: str) -> str:
    """Drop the dataset prefix and the generic ``individual_time_samples`` suffix."""
    parts = [p for p in PurePosixPath(folder).parts[1:] if p != "individual_time_samples"]
    return "/".join(parts) or folder


def cmd_labels(root: Path, dataset: str | None) -> None:
    """Print label counts aggregated per dataset."""
    per: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    where: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for s in subsets(root):
        info = s.info()
        ds, counts = str(info["dataset"]), info["label_counts"]
        assert isinstance(counts, dict)
        if dataset and ds != dataset:
            continue
        for label, n in counts.items():
            per[ds][label] += int(n)
            where[(ds, label)].append(s.folder)
    for ds, counts in sorted(per.items()):
        print(f"\n== {ds} ({sum(counts.values())} samples)")
        for label, n in counts.most_common():
            folders = sorted(set(where[(ds, label)]))
            shown = ", ".join(_short(f) for f in folders[:3])
            more = f" +{len(folders) - 3} more" if len(folders) > 3 else ""
            print(f"  {label:<28} {n:>7}   in {len(folders)} subset(s): {shown}{more}")


# ---------------------------------------------------------------------------- subset


def _describe_column(name: str, values: Sequence[object]) -> str:
    present = [v for v in values if v is not None]
    if not present:
        return f"  {name:<12} (all null)"
    if all(isinstance(v, int | float) for v in present):
        arr = np.asarray(present, dtype=np.float64)
        return (
            f"  {name:<12} {len(present)}/{len(values)} set, min {arr.min():g}, "
            f"median {np.median(arr):g}, max {arr.max():g}"
        )
    counts = collections.Counter(str(v) for v in present)
    top = ", ".join(f"{k} {v}" for k, v in counts.most_common(8))
    more = f" (+{len(counts) - 8} more)" if len(counts) > 8 else ""
    return f"  {name:<12} {len(counts)} distinct: {top}{more}"


def _signal_stats(h5: h5py.File, key: str, rows: npt.NDArray[np.int64]) -> str:
    data = np.asarray(h5[key][rows], dtype=np.float64).reshape(len(rows), -1)
    std = data.std(axis=1)
    flat = int(np.sum(std == 0))
    return (
        f"  {key:<34} {h5[key].dtype.name:>8} {tuple(h5[key].shape)!s:>22}  "
        f"mean {data.mean():.4g}, row std median {np.median(std):.4g}, "
        f"min {data.min():.4g}, max {data.max():.4g}, constant rows {flat}/{len(rows)}"
    )


def cmd_subset(root: Path, folder: str, stats_rows: int, seed: int) -> None:
    """Print metadata summaries and signal statistics for one subset."""
    s = find_subset(root, folder)
    info = s.info()
    print(f"== {s.folder}  [{info['dataset']}, {info['family']}, {info['representation']}]")
    print(
        f"  samples {info['num_samples']}, "
        f"audio {info['audio_sr']} Hz, seismic {info['seismic_sr']} Hz"
    )
    if info.get("note"):
        print(f"  note: {info['note']}")
    if info["name_label_mismatches"]:
        print(f"  filename/label mismatches: {info['name_label_mismatches']}")
    table = s.manifest(["labels", "distances_m", "node", "run", "clip", "scene", "offset_ms"])
    print("\nmetadata")
    print(_describe_column("labels", ["+".join(v or []) for v in table["labels"].to_pylist()]))
    dist = [d for ds in table["distances_m"].to_pylist() for d in ds or []]
    print(_describe_column("distance_m", dist))
    for name in ("node", "run", "clip", "scene", "offset_ms"):
        print(_describe_column(name, table[name].to_pylist()))
    n = len(table)
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(n, size=min(stats_rows, n), replace=False)).astype(np.int64)
    print(f"\nsignals (statistics over {len(rows)} random rows, values as stored)")
    with h5py.File(s.path / SIGNALS_FILE, "r") as h5:
        keys: list[str] = []
        h5.visititems(
            lambda name, obj: keys.append(name) if isinstance(obj, h5py.Dataset) else None
        )
        for key in keys:
            if key.startswith("signals/"):
                print(_signal_stats(h5, key, rows))
            else:
                print(f"  {key:<34} {h5[key].dtype.name:>8} {tuple(h5[key].shape)!s:>22}  (extra)")


# ---------------------------------------------------------------------------- sample


def _load_sample(root: Path, sid: str) -> tuple[Subset, dict[str, object]]:
    folder, _, _ = sid.strip("/").rpartition("/")
    s = find_subset(root, folder)
    table = s.manifest()
    hit = table.filter(pc.equal(table["sample_id"], pa.scalar(sid.strip("/"))))
    if len(hit) != 1:
        raise SystemExit(f"no sample {sid!r} in {s.folder} (ids are '<folder>/<file stem>')")
    row: dict[str, object] = hit.to_pylist()[0]
    return s, row


def _tree(obj: object, indent: str = "  ") -> list[str]:
    """Compact type/shape outline of a decoded original sample."""
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            sub = _tree(v, indent + "  ")
            if len(sub) == 1 and not isinstance(v, dict):
                out.append(f"{indent}{k}: {sub[0].strip()}")
            else:
                out += [f"{indent}{k}:", *sub]
        return out
    if isinstance(obj, torch.Tensor | np.ndarray) and obj.ndim > 0:
        return [f"{indent}{type(obj).__name__} {tuple(obj.shape)} {obj.dtype}"]
    text = repr(obj)
    return [f"{indent}{type(obj).__name__} {text[:100]}{'...' if len(text) > 100 else ''}"]


def cmd_sample(root: Path, sid: str, plot: Path | None, original: bool) -> None:
    """Print one sample's metadata and signals; optionally show its original layout and plot."""
    s, row = _load_sample(root, sid)
    meta = json.loads(str(row.pop("original_meta")))
    index = int(_num(row["sample_index"]))
    print(f"== {row['sample_id']}  (row {index} of {s.folder})")
    for k, v in row.items():
        if v is not None and k not in ("sample_id", "sample_index"):
            print(f"  {k:<15} {v}")
    with h5py.File(s.path / SIGNALS_FILE, "r") as h5:
        signals = {m: np.asarray(h5[f"signals/{m}"][index]) for m in MODALITIES}
        print("\nsignals (as stored)")
        for m, arr in signals.items():
            sr = _num(row[f"{m}_sr"])
            dur = f" = {arr.shape[-1] / sr:g} s" if row["representation"] == "waveform" else ""
            shape = f"{arr.shape}{dur}"
            print(
                f"  {m:<8} {arr.dtype.name:>8} {shape:<20}  mean {arr.mean():.4g}, "
                f"std {arr.std():.4g}, min {arr.min():.4g}, max {arr.max():.4g}"
            )
        if original:
            rebuilt = codec.decode(meta, lambda key, r: np.asarray(h5[key][r]))
            print("\noriginal sample structure (as torch.load returns it)")
            print("\n".join(_tree(rebuilt)))
    if plot is not None:
        plot_sample(signals, row, plot)
        print(f"\nplot written to {plot}")


# ---------------------------------------------------------------------------- plotting


def _style_axis(ax: object, title: str, xlabel: str, ylabel: str, grid: bool = True) -> None:
    from matplotlib.axes import Axes

    assert isinstance(ax, Axes)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=TEXT_PRIMARY)
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    if grid:
        ax.grid(color=GRID, linewidth=0.6)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_sample(
    signals: dict[str, npt.NDArray[np.generic]], row: dict[str, object], path: Path
) -> None:
    """Waveform + spectrogram per modality (or spectrum magnitude for fft subsets) to a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), facecolor=SURFACE, layout="constrained")
    labels = "+".join(str(v) for v in row["labels"]) if isinstance(row["labels"], list) else ""
    fig.suptitle(f"{row['sample_id']}   labels: {labels or '-'}", fontsize=10, color=TEXT_PRIMARY)
    for i, m in enumerate(MODALITIES):
        sr, arr = _num(row[f"{m}_sr"]), signals[m].astype(np.float64)
        if row["representation"] == "fft":
            # [re/im, interval, bin] -> log magnitude over the positive-frequency half
            mag = np.log10(np.hypot(arr[0], arr[1])[:, : arr.shape[-1] // 2] + 1.0)
            axes[i][0].imshow(mag.T, aspect="auto", origin="lower", cmap=SEQUENTIAL_CMAP)
            _style_axis(axes[i][0], f"{m}: log10 |FFT| per interval", "interval", "bin", False)
            axes[i][1].plot(mag.mean(axis=0), color=SERIES[m], linewidth=1)
            _style_axis(axes[i][1], f"{m}: mean log10 |FFT|", "bin", "log10 magnitude")
            continue
        t = np.arange(arr.size) / sr
        axes[i][0].plot(t, arr, color=SERIES[m], linewidth=0.6)
        _style_axis(axes[i][0], f"{m} waveform ({sr:g} Hz)", "time (s)", "raw value")
        nfft = 256 if arr.size >= 2048 else 32
        with np.errstate(divide="ignore"):  # silent spans give log10(0) = -inf, drawn blank
            axes[i][1].specgram(
                arr - arr.mean(), NFFT=nfft, Fs=sr, noverlap=nfft // 2, cmap=SEQUENTIAL_CMAP
            )
        _style_axis(
            axes[i][1], f"{m} spectrogram (mean removed)", "time (s)", "frequency (Hz)", False
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------- find


def cmd_find(root: Path, args: argparse.Namespace) -> None:
    """Search manifests and print matches per subset."""
    total = 0
    for s in subsets(root):
        if args.dataset and s.info()["dataset"] != args.dataset:
            continue
        t = s.manifest(["sample_id", "labels", "node", "run"])
        mask = np.ones(len(t), dtype=np.bool_)
        if args.label:
            mask &= np.array(
                [args.label in (v or []) for v in t["labels"].to_pylist()], dtype=np.bool_
            )
        if args.node:
            mask &= np.array([v == args.node for v in t["node"].to_pylist()], dtype=np.bool_)
        if args.run is not None:
            mask &= np.array([v == args.run for v in t["run"].to_pylist()], dtype=np.bool_)
        hits = t.filter(pa.array(mask))
        if len(hits):
            total += len(hits)
            print(f"{s.folder:<66} {len(hits):>7}   e.g. {hits['sample_id'][0]}")
    print(f"\n{total} matching samples")


# ---------------------------------------------------------------------------- main


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--root", type=Path, default=Path("datasets_unified"), help="unified dataset root"
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("overview", help="one line per subset")
    lab = sub.add_parser("labels", help="label counts per dataset")
    lab.add_argument("--dataset", help="only this dataset (e.g. M3N-VC)")
    sp = sub.add_parser("subset", help="details of one subset")
    sp.add_argument("folder", help="subset folder, e.g. M3N-VC/a06/individual_time_samples")
    sp.add_argument("--stats-rows", type=int, default=STATS_ROWS, help="rows sampled for stats")
    sp.add_argument("--seed", type=int, default=0)
    sm = sub.add_parser("sample", help="one sample")
    sm.add_argument("sample_id", help="'<folder>/<file stem>', as in the manifest")
    sm.add_argument("--plot", type=Path, help="write waveform/spectrogram PNG here")
    sm.add_argument("--original", action="store_true", help="show the rebuilt original structure")
    fd = sub.add_parser("find", help="search samples")
    fd.add_argument("--label")
    fd.add_argument("--node")
    fd.add_argument("--run", type=int)
    fd.add_argument("--dataset")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not (args.root / CATALOG_FILE).exists():
        print(
            f"{args.root} has no {CATALOG_FILE}; run scripts.preprocess.convert first",
            file=sys.stderr,
        )
        return 1
    if args.cmd == "overview":
        cmd_overview(args.root)
    elif args.cmd == "labels":
        cmd_labels(args.root, args.dataset)
    elif args.cmd == "subset":
        cmd_subset(args.root, args.folder, args.stats_rows, args.seed)
    elif args.cmd == "sample":
        cmd_sample(args.root, args.sample_id, args.plot, args.original)
    else:
        cmd_find(args.root, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
