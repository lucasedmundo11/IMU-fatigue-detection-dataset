# 4TU running-fatigue dataset — reading notes

Dataset: doi:10.4121/14307743.v1 — Marotta et al., *Sensors* 2021;21(10):3451.
31 `.mat` files, ~5.6 GB, all **MATLAB v7 (MATLAB 5.0)**, so `scipy.io.loadmat`
reads everything. No `mat73` or `h5py` needed. Peak RAM ≈ 200 MB per file.

## Code

Data processing only. Feature extraction and modelling come later.

| module | purpose |
|---|---|
| `src/dataset.py` | the reader — recordings, strides, subject map, lap segmentation |
| `src/tablefeats.py` | decodes `TableFeats.mat` (a MATLAB `table`) to a DataFrame |
| `src/mat5.py` | minimal MAT5 element reader used by `tablefeats` |
| `src/validate.py` | checks the data against the README and against our own assumptions |
| `src/__main__.py` | the CLI |

```bash
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python -m src inspect            # recordings, channels, stride counts
.venv/bin/python -m src validate --full    # README audit + subject-map re-derivation
.venv/bin/python -m src export-tablefeats  # -> data/preprocessed/TableFeats.csv
```

Run from the project root. Verified on the project venv — Python 3.13.11,
numpy 2.5.2, scipy 1.18.0, pandas 3.0.5 — and on system Python 3.9 with
numpy 1.x / pandas 1.x, with identical output on both.

`pyarrow` is not installed, so `--out *.parquet` falls back to csv. Uncomment it
in `requirements.txt` if you want parquet.

As a library:

```python
import src

for rec in src.iter_recordings():
    acc  = rec.acc("PEL")        # (N, 3) m/s^2
    knee = rec.angle("RKNE")     # (N, 3) DEGREES, not radians
    keep = rec.paper_mask()      # laps 2-4 / 8-10 / 11-13
```

Paths resolve by content, not by name: `find_data_dir()` searches `data/raw`
then `data/` for a directory that actually holds the recordings, so moving or
renaming the dataset folder does not break anything. `$FATIGUE_DATA` overrides.

`validate` separates **known** defects in the published dataset from
**unexpected** ones and exits non-zero only for the latter, so it works as a
regression check.

## Where the README is wrong

Verified by `python3 -m src validate` (2 known defects, 3 omissions).

1. **Joint angles are DEGREES, not radians.** The README says
   `segment.XXX.angle: angle data for a given joint (rad)`. Right-knee flexion
   peaks at 109.3 — that is degrees; as radians it would be 6262°. Segment
   `acc` (m/s², pelvis median |acc| 13.5) and `angvel` (rad/s, foot peak 17.1
   ≈ 977 °/s) *are* as documented.
2. **"Each of the 8 subjects has 4 files" is false.** `p007` has no
   `postfatigue1200m` recording — 31 files, not 32. Its *strides* file still
   contains that run (see below), so p007 yields only 2 of the 3 classes for
   any pipeline built on the raw recordings.
3. Undocumented fields: `segment.<S>` also carries `ori`, `pos`, `vel`,
   `angacc`; `joint.<J>` also carries `angle_XZY`. The stride-file contents are
   not described at all.

## Structural gotchas

- **The `CW`/`CCW` token is not positional.** For `p003`, `p006`, `p008` the
  0-2k half was run CW and the 2-4k half CCW — the reverse of the other five.
  Parse the run token, not the position. A pattern like
  `p\d+_HDSL_(0-2k|2-4k|postfatigue1200m)` silently matches only the 7
  postfatigue files and drops all 16 CW/CCW recordings.
- **Two different top-level layouts.** Most files expose
  `COM`/`info`/`joint`/`segment` at the top level; some (e.g.
  `p001_HDSL_postfatigue1200m`) nest all four inside one struct named after the
  trial. `dataset._unwrap` normalises both.
- **Don't auto-pick the largest numeric array.** A size heuristic lands on the
  orientation quaternions or positions, not on the acc/angvel/angle signals the
  study is about. `dataset.py` selects channels explicitly (69 total).

## Subject IDs

`TableFeats` numbers subjects `{1, 5, 6, 13, 15, 16, 17, 18}`; the recordings
are `p001…p008`. They map **in order**:

```
p001→1  p002→5  p003→6  p004→13  p005→15  p006→16  p007→17  p008→18
```

Established via stature: MVN scales its body model to the entered subject
height, and the calibration T-pose reproduces every TableFeats height to within
0.2 cm (r = 1.0000). Speed ranking alone agrees 6/8 and cannot separate
p006/p008, whose TableFeats speeds differ by 0.01 km/h; stature separates them
cleanly. Re-derivable with `python3 -m src validate --full`.

## What the stride files contain

`pXXX_strides.mat` holds 45 channels, each `(150 time-normalised points ×
n_strides)`: 7 joints × {`X`, `Y`, resultant} plus 8 body segments ×
{`nacc`, `jerk`, `angvel`}.

They cover **all three runs concatenated chronologically**, not just the 4000 m
run. Measured gait-cycle period from foot angular velocity is 0.68–0.77 s
(autocorrelation peaks 0.86–0.96). The 4000 m-only hypothesis would require
0.57–0.63 s/stride, i.e. *more* strides than the recordings can produce —
impossible, since segmentation only drops strides. Including the postfatigue
run predicts counts 3.5–14.6 % above actual, the expected direction given
dropped turnaround and incomplete cycles. `p007` confirms it: without a
postfatigue file it falls 19 % short, and adding an inferred ~620-stride
postfatigue run brings it to −6 %, in line with everyone else.
(Periods were measured by autocorrelation of foot angular velocity against
per-run durations; that one-off probe is gone now the conclusion is
documented on `Strides`.)

**Caveat:** the stride arrays carry no run or lap index, so strides cannot be
labelled directly. Use the recordings when provenance matters — there
`Recording.lap_index()` and `.paper_mask()` give unambiguous labels.

## Labels

13 laps total: 10 in the 4000 m run, 3 in the postfatigue run. The paper samples
three laps per class:

| laps | class | file |
|---|---|---|
| 2–4 | `none` (0) | `0-2k` |
| 8–10 | `mild` (1) | `2-4k` |
| 11–13 | `heavy` (2) | `postfatigue1200m` |

`mild` is genuine accumulated fatigue over the 4000 m run — the fatiguing
protocol itself (run 2) is **not distributed**.

Confirmed against the decoded `TableFeats`: 3 classes, exactly balanced per
subject (e.g. 432/432/432), mean RPE **9.5 / 11.5 / 14.1**.

Lap boundaries are cut on *fractional* cumulative distance from `COM.trial`, not
on a hard 400 m mark — MVN dead-reckoning undershoots by a few percent (tracked
path 1832–1979 m over a nominal 2000 m). Winding number confirms 5.0 laps per
2 km half.

## TableFeats.mat

A MATLAB `table`, i.e. an MCOS classdef object, which scipy returns as an opaque
stub. The payload lives in `__function_workspace__`, a raw MAT5 element stream;
`src/mat5.py` walks it directly.

**12,513 rows × 169 columns.** 165 are numeric doubles; 4 are categorical char
arrays (`footstrike`, `dir`, `gender`, `d_l`) and are *not* recovered — they are
skipped when aligning names to columns.

Key columns: `rpe` (8–19), `sub`, `fatigue` (0/1/2 label), `speed`, `HR_norm`,
`weight`, `height`, `experience`, then gait-event angles (`IC_*`, `MS_max_*`,
`TO_min_*`, …) and per-segment acc/angvel moments. Values are already z-scored
**per subject**, as the README states.

## Baseline result (superseded — code removed)

Recorded here only so the number is not lost; the ML pipeline was removed when
`src/` was trimmed to data processing.

Windowed features (10 s window, 5 s hop) over 69 channels x 6 descriptors =
414 features, 1917 windows, leave-one-subject-out random forest:

| normalisation | pooled acc | macro F1 |
|---|---|---|
| none | 0.436 | 0.429 |
| **per subject** | **0.694** | **0.701** |

Chance is 0.333. Per-subject z-scoring is essential — between-subject spread in
speed (9.4–13.7 km/h) and stature (164–188 cm) otherwise swamps the
within-subject fatigue effect. Confusion was almost entirely between *adjacent*
classes; `none`↔`heavy` errors numbered 1 and 12.

Two caveats for whatever replaces it: per-subject normalisation uses the
held-out subject's own distribution (transductive — per-wearer calibration, no
labels, but it assumes a baseline exists at test time), and `p007` scores 0.489
because it contributes only 2 of the 3 classes.
