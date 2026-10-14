# Unified dataset format

These scripts convert every sample folder under `datasets/` into one common format in `datasets_unified/`, and prove the conversion changed nothing except the format. `datasets/` is only ever read.

## Quick start

```bash
uv run python -m scripts.preprocess.convert       --src datasets --out datasets_unified --workers 48
uv run python -m scripts.preprocess.verify        --src datasets --out datasets_unified --workers 48
uv run python -m scripts.preprocess.check_derived --src datasets --out datasets_unified --workers 48
```

- `convert` is resumable: finished subsets are skipped (`--overwrite` redoes them), and each subset is built in `<folder>.partial/` and renamed only once complete. `--only 'M3N-VC/*'` restricts it to matching folders.
- `verify` exits 0 only if every check passes and writes `datasets_unified/verify_report.json`.
- `inspect_unified` looks at the result (read-only), see [Inspecting](#inspecting).
- `check_derived` covers the frequency-domain folders that were deliberately not converted (see below), and writes `derived_report.json`.

## Layout

```
datasets_unified/
  catalog.json                  # converted subsets, excluded folders (+ reason), files not converted, split summary
  <source folder>/              # same relative path as in datasets/, e.g. M3N-VC/a06/individual_time_samples
    signals.h5
    manifest.parquet
    info.json                   # sample rates, shapes/dtypes, label and node counts, notes
  splits/<index file>.parquet   # every datasets/**/*.txt index file, resolved to sample ids
```

Each original folder becomes one subset, with files in natural filename order (`clip_2` before `clip_10`). Row `i` of `manifest.parquet` is row `i` of every `signals/*` array.

### `signals.h5`

| Dataset | Shape | Contents |
|---|---|---|
| `signals/audio` | `[N, L_audio]` | Audio waveform of each 2 s (sometimes 4 s) clip |
| `signals/seismic` | `[N, L_seismic]` | Seismic waveform |
| `extras/...` | `[M, ...]` | Additional arrays from the source, currently only Parkland `*_dwt.pt` wavelet coefficients (`extras/wavelet/{db4,sym8}/shake/{audio,seismic}`) |

- **Values and dtypes are exactly the source's.** No resampling, no normalization, no casting: int16 stays int16, int64 stays int64, float64 stays float64. That's also why dtypes differ between subsets (see `info.json` → `columns`).
- **Waveforms are flattened.** For example, the source's `[1, 10, 1600]` (10 intervals of 0.2 s) becomes `[16000]`. This is a pure reshape in time order, and the original shape is kept in `original_meta`.
- `Parkland/individual_samples` is the one frequency-domain subset (`representation = "fft"`). It keeps the source shape `[N, 2, 10, 1600]` (real/imag × interval × bin), because about 17k of its samples exist in no time-domain folder.
- The HDF5 file attributes, and the `audio_sr`/`seismic_sr` columns, hold the sample rates.

### `manifest.parquet`

| Column | Meaning |
|---|---|
| `sample_index`, `sample_id` | Row number, and the stable id `<source folder>/<filename without .pt>` |
| `dataset`, `subset`, `representation` | `Parkland` / `GracesQuarters` / `ICT` / `WestPoint` / `M3N-VC`; source folder; `waveform` or `fft` |
| `labels` | Class names, lowercase, in source order (multiple for multi-vehicle clips) |
| `distances_m` | Distance (m) per label, aligned with `labels`; null where unknown or a sentinel in the source (-1, NaN, inf) |
| `name_labels` | Class names implied by the filename (for cross-checking; null if the filename carries none) |
| `node`, `run`, `clip` | Sensor node (`rs3`, `gq-1`, …), run number, clip index (consecutive clips are consecutive in time) |
| `recording`, `scene`, `offset_ms` | Parkland recording folder; M3N-VC scene; M3N-VC clip offset (`o1000` clips overlap the plain clips by half) |
| `node_xy`, `vehicle_xy`, `clip_start` | M3N-VC positions (local metres) and clip start (unix s) |
| `audio_sr`, `seismic_sr` | Sample rates (Hz) |
| `source_file`, `source_sha256`, `source_bytes` | The original file and its hash at conversion time |
| `original_meta` | Everything in the original file except the big arrays, as lossless tagged JSON (types such as `np.float64`, sets and object arrays are preserved). Array positions are refs to `signals.h5` rows |

The normalized columns are conveniences; `original_meta` + `signals.h5` are the ground truth.

### Reading

```python
import h5py, pyarrow.parquet as pq

root = "datasets_unified/M3N-VC/a06/individual_time_samples"
rows = pq.read_table(f"{root}/manifest.parquet").to_pylist()
i = 0
with h5py.File(f"{root}/signals.h5", "r") as h5:
    audio = h5["signals/audio"][i]  # 1-D waveform, source dtype
    seismic = h5["signals/seismic"][i]
labels, rate = rows[i]["labels"], rows[i]["audio_sr"]
```

To rebuild the original object exactly (same structure as `torch.load` on the source), use `scripts.preprocess.codec.decode(json.loads(original_meta), lambda key, row: h5[key][row])`.

## Inspecting

```bash
I="uv run python -m scripts.preprocess.inspect_unified"
$I overview                                   # one line per subset: N, shapes, dtypes, rates, top labels
$I labels --dataset GracesQuarters            # label counts per dataset and which subsets hold them
$I subset M3N-VC/a06/individual_time_samples  # metadata distributions + signal stats (1000 random rows)
$I sample M3N-VC/a06/individual_time_samples/a06_cx30_run6_rs1_00000_9 --original --plot s.png
$I find --label cx30 --node rs1               # matching samples per subset
```

`sample --original` prints the structure exactly as `torch.load` returned it from the source file; `--plot` writes waveform + spectrogram per modality (log-magnitude spectra for the `fft` subset). Use `--root` for a unified dataset elsewhere.

## What `verify` proves

For every source folder and every source file, checked against the files as they are on disk now:

1. Every folder containing `.pt` files is either converted or explicitly excluded; an unknown new folder fails.
2. The manifest lists exactly the files on disk, each once, in order.
3. Each source file's SHA-256 still matches the hash recorded at conversion.
4. Each source sample, freshly `torch.load`ed, equals the object rebuilt from `signals.h5` + `original_meta` under a strict comparison: identical Python types (`float` ≠ `np.float64`, `list` ≠ `tuple`), dict key order, dtypes, shapes and raw bytes (so NaN and `-0.0` are compared bit for bit).
5. As an independent check that doesn't go through the metadata decoder, the raw bytes of every `data[node][modality]` array equal the stored HDF5 row.
6. The normalized columns equal a fresh normalization of the source.
7. HDF5 holds no unreferenced arrays and no unexpected datasets.
8. Every non-empty line of every index file is present, in order, and every resolved sample id points to a file with the same folder and name.

The test suite (`tests/scripts/preprocess/test_pipeline.py`) confirms that a single flipped bit, an edited label, a changed or deleted source file, an unregistered folder, or a tampered split each make `verify` fail.

## What was not converted, and caveats

- **Frequency-domain folders** `Parkland/individual_freq_samples`, `Parkland/individual_freq_samples_with_wind` and `Parkland/extra_freq_samples`: each file is the per-interval FFT of the same-named time-domain file. `check_derived` proves that for every file (same label and flags; spectra match the FFT of the unified waveform up to float32 rounding, since the source computed its FFTs before casting to float32).
- **Non-sample files** (`catalog.json` → `not_converted_files`): the M3N-VC raw recordings in `M3N-VC/zenodo/` (continuous parquet streams plus the downloaded zips). They're already in an open format, and the clips cut from them *are* converted.
- **Overlapping folders are kept as-is**: the GracesQuarters `*Filtered*` / `*_random` folders are subsets of `day_{1,2}_sensor_all` (same bytes); `2024-08-07-GQ_past` ⊂ `2024-08-07-GQ`; M3N-VC `o1000` clips overlap the plain clips by half. Deduplicate (e.g. on `source_sha256` or the signals) before mixing subsets for training.
- **Labels needing care**:
  - Parkland: `labels` comes from the recording name in the filename (`forester_rs1_3.pt` → `forester`). In the time-domain folders this matches the stored ids exactly, but 16,907 legacy samples in `individual_samples` store ids from an incompatible scheme (e.g. `forester` and `mustang` both stored as 3).
  - GracesQuarters `*FilteredNone*`: every clip is labeled `background` in the source on purpose (vehicle-free stretches of vehicle runs), so its filename prefix (the run's vehicle) disagrees with `labels`. These are counted in `name_label_mismatches`, which is expected here.
  - `GracesQuartersMixed`: the stored integer labels disagree with the filenames (5 of 6 vehicle pairs store `(0, 1)`). The `labels` column comes from the filename; the raw ids are preserved in `original_meta`.
  - Integer labels elsewhere are mapped with the tables in `specs.py`: Parkland from `cs537aiot/.../extract_samples.py`, GQ and ICT-mixed inferred from filenames. Any sample whose filename disagrees with its label is counted in `info.json` → `name_label_mismatches`.
  - Spellings are kept from the source (Parkland `warhog` vs GQ `warthog`, GQ `truck` / `sedan` / `silverado`); only lowercasing is applied.
  - GQ `day_*_sensor_all` files also carry `targets` (e.g. `['husky', 'pedestrian']`) and `type`. They're in `original_meta`, not in `labels`.
- **Source label quirks kept as-is**: `polairs` (typo, 2 samples in `2024-08-05-GQ`) and repeated labels such as `warthog+warthog` (556 samples in `2024-08-06-GQ`).
- **M3N-VC `a06` seismic gaps**: about 8% of `a06` clips contain seismic stretches of 0.1 s or more that are exactly linear or constant, i.e. gaps in the raw recording filled in by the M3N-VC preprocessing (the raw geophone stream runs at ~197.9 Hz instead of 200). The other scenes and datasets show none in a 500-clip check. Consider masking or dropping those clips.
- **Sample rates**: taken from the preprocessing code (8000/100 Hz) and the M3N-VC raw data (1600/200 Hz). For `ict_pt_files` (32000/400 samples) the rate isn't recorded anywhere; 8000/100 Hz (4 s) is inferred (see `info.json` → `note`).
- **Splits**: index files point to paths on other machines. Lines are resolved by (top-level folder, parent folder, filename). Some index files refer to data that isn't in `datasets/` (e.g. `Parkland_1107`, `Parking_Lot`); those lines keep `sample_id = null` and are counted in `catalog.json`.
