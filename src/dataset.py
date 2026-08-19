"""Reader for the 4TU running-fatigue dataset (doi:10.4121/14307743.v1).

Marotta et al., "Fatigue detection in running with IMUs", Sensors 2021;21:3451.

Everything here was verified against the shipped README and against the data
itself; where the two disagree the data wins and the discrepancy is noted.

Layout
------
p<NNN>_HDSL_<CW|CCW>_<0-2k|2-4k>.mat   first  run, 4000 m, 10 laps, constant speed
p<NNN>_HDSL_postfatigue1200m.mat       third  run, 1200 m,  3 laps, same speed
p<NNN>_strides.mat                     time-normalised strides, all runs
TableFeats.mat                         the authors' normalised ML feature table

The second run -- the fatiguing protocol itself -- is NOT distributed.

README corrections
------------------
* joint.<J>.angle is in DEGREES, not radians. Knee flexion peaks near 110; in
  radians that would be 6262 deg. segment.acc (m/s^2) and segment.angvel
  (rad/s) are as documented.
* "Each of the 8 subjects has 4 files" is false: p007 has no postfatigue1200m
  recording. Its strides file still contains that run.
* segment.<S> also carries ori/pos/vel/angacc, and joint.<J> also carries
  angle_XZY; the README lists only acc/angvel and angle.
* The CW/CCW token is not positional -- for p003, p006 and p008 the 0-2k half
  was run CW and the 2-4k half CCW, the reverse of the others. Parse the run
  token, never the field order.

Condition labels
----------------
The 4000 m run is 10 laps; the postfatigue run adds laps 11-13. The paper
samples three laps per class:

    laps  2-4   no fatigue     -> 0-2k file
    laps  8-10  mild fatigue   -> 2-4k file
    laps 11-13  heavy fatigue  -> postfatigue file

`mild` is genuine accumulated fatigue over the 4000 m run. Reproduced from the
decoded TableFeats: 3 classes, exactly balanced per subject, mean RPE
9.5 / 11.5 / 14.1.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io as sio

# --------------------------------------------------------------------------
# Constants verified against the data
# --------------------------------------------------------------------------

FS = 240.0  # info.frameRate, identical in all 31 files

SEGMENTS = ("STE", "PEL", "RUL", "LUL", "RLL", "LLL", "RFO", "LFO")
JOINTS = ("L5S1", "RHIP", "LHIP", "RKNE", "LKNE", "RANK", "LANK")

SEGMENT_LABEL = {
    "STE": "sternum", "PEL": "pelvis",
    "RUL": "right thigh", "LUL": "left thigh",
    "RLL": "right tibia", "LLL": "left tibia",
    "RFO": "right foot", "LFO": "left foot",
}
JOINT_LABEL = {
    "L5S1": "trunk", "RHIP": "right hip", "LHIP": "left hip",
    "RKNE": "right knee", "LKNE": "left knee",
    "RANK": "right ankle", "LANK": "left ankle",
}

# run token -> (condition, class code, laps spanned, laps the paper uses)
RUNS = {
    "0-2k":             ("none",  0, (1, 2, 3, 4, 5),  (2, 3, 4)),
    "2-4k":             ("mild",  1, (6, 7, 8, 9, 10), (8, 9, 10)),
    "postfatigue1200m": ("heavy", 2, (11, 12, 13),     (11, 12, 13)),
}

# TableFeats numbers subjects {1,5,6,13,15,16,17,18}; the recordings are
# p001..p008. They map in order. Established by stature: MVN scales its body
# model to the entered subject height, and the calibration T-pose reproduces
# every TableFeats height to within 0.2 cm (r = 1.0000). Speed ranking alone
# agrees 6/8 and cannot separate p006/p008, whose TableFeats speeds differ by
# 0.01 km/h. Re-checkable with `python3 -m src validate`.
SUBJECT_ID = {
    "p001": 1, "p002": 5, "p003": 6, "p004": 13,
    "p005": 15, "p006": 16, "p007": 17, "p008": 18,
}

# 45 stride channels: 7 joints x {X, Y, resultant} + 8 bodies x {nacc, jerk,
# angvel}. Undocumented in the README.
STRIDE_JOINTS = ("lankle", "rankle", "lknee", "rknee", "lhip", "rhip", "trunk")
STRIDE_BODIES = ("lfoot", "rfoot", "lll", "rll", "lul", "rul",
                 "pelvis", "sternum")

RECORDING_RE = re.compile(
    r"^p(?P<subject>\d{3})_HDSL"
    r"(?:_(?P<direction>CW|CCW))?"
    r"_(?P<run>0-2k|2-4k|postfatigue1200m)$",
    re.IGNORECASE,
)
STRIDES_RE = re.compile(r"^p(?P<subject>\d{3})_strides$", re.IGNORECASE)


def find_data_dir(start: str | Path | None = None) -> Path:
    """Locate the directory holding the .mat files.

    Checked in order: $FATIGUE_DATA, then any directory at or below
    data/raw and data/ that actually contains the recordings. Resolving by
    content rather than by a hard-coded name keeps this working when the
    dataset folder is moved or renamed.
    """
    if start is not None:
        return Path(start)

    env = os.environ.get("FATIGUE_DATA")
    if env:
        return Path(env)

    roots = [Path("data/raw"), Path("data"), Path(".")]
    for root in roots:
        if not root.is_dir():
            continue
        for cand in [root, *sorted(p for p in root.iterdir() if p.is_dir())]:
            if any(RECORDING_RE.match(p.stem) for p in cand.glob("*.mat")):
                return cand
    return Path("data/raw")


# Where derived tables are written. `data/preprocessed` is the convention in
# this project; raw inputs stay untouched under `data/raw`.
PREPROCESSED = Path("data/preprocessed")


# --------------------------------------------------------------------------
# Low-level loading
# --------------------------------------------------------------------------

def _loadmat(path: Path) -> dict:
    """Load a v7 .mat. Every file in this dataset is v7 (MATLAB 5.0), so no
    mat73 / h5py fallback is needed."""
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


def _unwrap(raw: dict) -> dict:
    """Normalise the two top-level layouts used in this dataset.

    Most recordings expose COM/info/joint/segment at the top level, but some
    (e.g. p001_HDSL_postfatigue1200m) nest all four inside a single struct
    named after the trial.
    """
    top = [k for k in raw if not k.startswith("__")]
    if "segment" in top:
        return {k: raw[k] for k in top}
    if len(top) == 1 and hasattr(raw[top[0]], "_fieldnames"):
        node = raw[top[0]]
        return {f: getattr(node, f) for f in node._fieldnames}
    return {k: raw[k] for k in top}


# --------------------------------------------------------------------------
# Recordings
# --------------------------------------------------------------------------

@dataclass
class Recording:
    """One continuous MVN export."""

    subject: str            # "p001"
    sub_id: int             # TableFeats `sub`
    run: str                # "0-2k" | "2-4k" | "postfatigue1200m"
    direction: str | None   # "CW" | "CCW" | None
    condition: str          # "none" | "mild" | "heavy"
    label: int              # 0 | 1 | 2
    fs: float
    segment: object
    joint: object
    com: np.ndarray         # (N, 3) centre-of-mass trajectory
    info: object

    @property
    def n_samples(self) -> int:
        return self.com.shape[0]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs

    def acc(self, seg: str) -> np.ndarray:
        """Segment acceleration, m/s^2, shape (N, 3)."""
        return getattr(self.segment, seg).acc

    def angvel(self, seg: str) -> np.ndarray:
        """Segment angular velocity, rad/s, shape (N, 3)."""
        return getattr(self.segment, seg).angvel

    def angle(self, joint: str) -> np.ndarray:
        """Joint angle, DEGREES (the README says radians -- it is wrong)."""
        return getattr(self.joint, joint).angle

    def distance(self) -> np.ndarray:
        """Cumulative horizontal distance travelled, metres, shape (N,)."""
        step = np.linalg.norm(np.diff(self.com[:, :2], axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(step)])

    def lap_index(self) -> np.ndarray:
        """Absolute lap number (1-13) for every sample.

        MVN position is dead-reckoned and undershoots true distance by a few
        percent (1832-1979 m over a nominal 2000 m), so laps are cut on
        *fractional* distance rather than a hard 400 m boundary. Both runs held
        constant speed by design, which makes the split well conditioned.
        """
        laps = RUNS[self.run][2]
        d = self.distance()
        frac = d / max(d[-1], 1e-9)
        idx = np.clip((frac * len(laps)).astype(int), 0, len(laps) - 1)
        return np.asarray(laps, dtype=int)[idx]

    def paper_mask(self) -> np.ndarray:
        """True where a sample falls in a lap the paper actually uses."""
        return np.isin(self.lap_index(), list(RUNS[self.run][3]))

    def __repr__(self) -> str:
        return (f"<Recording {self.subject}(sub={self.sub_id}) {self.run} "
                f"{self.direction or '-'} {self.condition} "
                f"{self.n_samples:,} samples {self.duration_s:.0f}s>")


def load_recording(path: str | Path) -> Recording:
    path = Path(path)
    m = RECORDING_RE.match(path.stem)
    if not m:
        raise ValueError(f"not a recording filename: {path.name}")

    d = _unwrap(_loadmat(path))
    subject = f"p{m.group('subject')}"
    run = m.group("run").lower()
    condition, label, _, _ = RUNS[run]

    return Recording(
        subject=subject,
        sub_id=SUBJECT_ID[subject],
        run=run,
        direction=(m.group("direction") or "").upper() or None,
        condition=condition,
        label=label,
        fs=float(d["info"].frameRate),
        segment=d["segment"],
        joint=d["joint"],
        com=d["COM"].trial,
        info=d["info"],
    )


# --------------------------------------------------------------------------
# Strides
# --------------------------------------------------------------------------

@dataclass
class Strides:
    """Time-normalised strides for one subject.

    45 channels, each (150 normalised points x n_strides), covering all three
    runs concatenated chronologically -- including the postfatigue run for
    p007, whose raw recording is missing from the dataset.

    The arrays carry no run or lap index, so strides cannot be labelled
    directly; use the recordings when provenance matters.
    """

    subject: str
    sub_id: int
    channels: dict[str, np.ndarray]

    @property
    def n_strides(self) -> int:
        return next(iter(self.channels.values())).shape[1]

    @property
    def n_points(self) -> int:
        return next(iter(self.channels.values())).shape[0]

    def names(self) -> list[str]:
        return list(self.channels)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.channels[name]

    def __repr__(self) -> str:
        return (f"<Strides {self.subject}(sub={self.sub_id}) "
                f"{len(self.channels)} channels x {self.n_points} pts "
                f"x {self.n_strides} strides>")


def load_strides(path: str | Path) -> Strides:
    path = Path(path)
    m = STRIDES_RE.match(path.stem)
    if not m:
        raise ValueError(f"not a strides filename: {path.name}")

    raw = _loadmat(path)
    node = raw[[k for k in raw if not k.startswith("__")][0]]
    subject = f"p{m.group('subject')}"

    return Strides(
        subject=subject,
        sub_id=SUBJECT_ID[subject],
        channels={f: getattr(node, f) for f in node._fieldnames},
    )


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _index(data_dir: str) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    recs, strides = [], []
    for p in sorted(Path(data_dir).glob("*.mat")):
        if RECORDING_RE.match(p.stem):
            recs.append(p)
        elif STRIDES_RE.match(p.stem):
            strides.append(p)
    return tuple(recs), tuple(strides)


def recording_paths(data_dir: str | Path | None = None) -> list[Path]:
    return list(_index(str(find_data_dir(data_dir)))[0])


def strides_paths(data_dir: str | Path | None = None) -> list[Path]:
    return list(_index(str(find_data_dir(data_dir)))[1])


def subjects(data_dir: str | Path | None = None) -> list[str]:
    return sorted({f"p{RECORDING_RE.match(p.stem).group('subject')}"
                   for p in recording_paths(data_dir)})


def iter_recordings(data_dir: str | Path | None = None):
    """Yield each recording in turn. Loaded lazily -- each is 100-300 MB."""
    for p in recording_paths(data_dir):
        yield load_recording(p)


def iter_strides(data_dir: str | Path | None = None):
    for p in strides_paths(data_dir):
        yield load_strides(p)
