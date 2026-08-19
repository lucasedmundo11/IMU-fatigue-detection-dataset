"""Check the data against the dataset README and against our own assumptions.

Two groups of checks:

* `audit_readme` -- every factual claim the README makes about structure and
  units. Units are verified physically rather than taken on trust: gravity
  fixes the acceleration scale and known running kinematics fix the angular
  scales. Two claims fail; see NOTES.md.
* `verify_subject_map` -- the pXXX -> TableFeats `sub` mapping baked into
  dataset.SUBJECT_ID. This is an inference, not documented anywhere, so it is
  worth re-checking whenever the data or the mapping changes.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from . import dataset as ds
from .tablefeats import load_tablefeats

OK, WARN, FAIL = "  ok  ", " WARN ", " FAIL "


class Report:
    """Accumulates check results and prints them as they happen.

    Failures split two ways. Defects already known to be in the published
    dataset are expected and must not fail the run, or this module is useless
    as a regression check. Anything else is new and does fail.
    """

    def __init__(self):
        self.known = 0
        self.unexpected = 0
        self.warned = 0

    def add(self, level: str, claim: str, detail: str = "",
            known: bool = False) -> None:
        if level is FAIL:
            if known:
                self.known += 1
                level = "KNOWN "
            else:
                self.unexpected += 1
        elif level is WARN:
            self.warned += 1
        print(f"[{level}] {claim}")
        if detail:
            print(f"         {detail}")


def audit_readme(data_dir: Path, rep: Report) -> None:
    print("\n=== README structural claims ===\n")

    recordings = ds.recording_paths(data_dir)
    strides = ds.strides_paths(data_dir)
    subject_ids = sorted({re.match(r"p(\d+)", p.name).group(1)
                          for p in Path(data_dir).glob("p*.mat")})

    rep.add(OK if len(subject_ids) == 8 else FAIL, "8 subjects",
            f"found {len(subject_ids)}: {', '.join(subject_ids)}")

    want = {"0-2k", "2-4k", "postfatigue1200m", "strides"}
    missing = []
    for s in subject_ids:
        got = {re.search(r"(0-2k|2-4k|postfatigue1200m|strides)", p.name,
                         re.I).group(1).lower()
               for p in Path(data_dir).glob(f"p{s}_*.mat")}
        if got != want:
            missing.append((s, sorted(want - got)))
    rep.add(OK if not missing else FAIL, "each subject has 4 files",
            "; ".join(f"p{s} missing {m}" for s, m in missing)
            or f"{len(recordings)} recordings + {len(strides)} stride files",
            known=missing == [("007", ["postfatigue1200m"])])

    rec = ds.load_recording(recordings[0])

    seg_members = list(rec.segment._fieldnames)
    rep.add(OK if seg_members == list(ds.SEGMENTS) else FAIL,
            "segment members STE/PEL/RUL/LUL/RLL/LLL/RFO/LFO",
            f"got {seg_members}")

    joint_members = list(rec.joint._fieldnames)
    rep.add(OK if joint_members == list(ds.JOINTS) else FAIL,
            "joint members L5S1/RHIP/LHIP/RKNE/LKNE/RANK/LANK",
            f"got {joint_members}")

    seg_fields = list(getattr(rec.segment, ds.SEGMENTS[0])._fieldnames)
    rep.add(OK if {"acc", "angvel"} <= set(seg_fields) else FAIL,
            "segment.<S> has .acc and .angvel", f"fields: {seg_fields}")
    undocumented = [f for f in seg_fields if f not in ("acc", "angvel")]
    if undocumented:
        rep.add(WARN, "segment carries fields the README omits",
                f"undocumented: {undocumented}")

    joint_fields = list(getattr(rec.joint, ds.JOINTS[0])._fieldnames)
    rep.add(OK if "angle" in joint_fields else FAIL,
            "joint.<J> has .angle", f"fields: {joint_fields}")
    undocumented = [f for f in joint_fields if f != "angle"]
    if undocumented:
        rep.add(WARN, "joint carries fields the README omits",
                f"undocumented: {undocumented}")

    rep.add(OK if rec.fs == 240 else FAIL, "recorded at 240 Hz",
            f"frameRate={rec.fs:g}")
    rep.add(OK if "MVN" in str(rec.info.system) else WARN, "MVN Link system",
            f"{rec.info.system}, {rec.info.mvnVersion}")
    rep.add(OK if len(seg_members) == 8 else FAIL, "8 IMUs",
            f"{len(seg_members)} instrumented segments exported "
            f"(full body model has {rec.info.segments.count})")

    print("\n=== README unit claims (verified physically) ===\n")

    mag = np.linalg.norm(rec.acc("PEL"), axis=1)
    rep.add(OK if 5 < np.median(mag) < 40 else WARN, "segment.acc in m/s^2",
            f"pelvis |acc| median {np.median(mag):.2f}, "
            f"p99 {np.percentile(mag, 99):.1f} -- consistent with m/s^2 "
            f"(g=9.81), not g-units")

    # Foot angular velocity in running peaks around 600-1000 deg/s.
    peak_av = np.abs(rec.angvel("RFO")).max()
    rep.add(OK if 5 < peak_av < 60 else WARN, "segment.angvel in rad/s",
            f"right-foot peak {peak_av:.1f} = {np.degrees(peak_av):.0f} deg/s, "
            f"the expected magnitude for running")

    knee = rec.angle("RKNE")
    flex = knee[:, int(np.argmax(knee.max(0) - knee.min(0)))]
    worst = max(np.abs(rec.angle(j)).max() for j in ds.JOINTS)
    if flex.max() > 6.3:
        rep.add(FAIL, "joint.<J>.angle in rad -- README is WRONG",
                f"right-knee flexion peaks at {flex.max():.1f}; that is "
                f"degrees. As radians it would be {np.degrees(flex.max()):.0f} "
                f"deg, impossible. Max |angle| over all joints = {worst:.1f}.",
                known=True)
    else:
        rep.add(OK, "joint.<J>.angle in rad", f"peak {flex.max():.2f} rad")

    del rec

    print("\n=== stride files (undocumented in README) ===\n")
    s = ds.load_strides(strides[0])
    rep.add(WARN, "README does not describe stride-file contents",
            f"{len(s.channels)} channels, each ({s.n_points} normalised "
            f"points x n_strides); {s.subject} has {s.n_strides} strides")


def verify_subject_map(data_dir: Path, rep: Report) -> None:
    """Re-derive dataset.SUBJECT_ID from MVN's scaled body model.

    MVN scales its skeleton to the subject's entered body height, so the
    calibration T-pose encodes stature directly. TableFeats stores that same
    height, which makes it an exact fingerprint.
    """
    print("\n=== subject mapping (pXXX -> TableFeats `sub`) ===\n")

    rows = []
    for path in ds.recording_paths(data_dir):
        m = ds.RECORDING_RE.match(path.stem)
        if m.group("run").lower() != "0-2k":
            continue
        raw = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        info = ds._unwrap(raw)["info"]
        pos = np.asarray(info.calibration.Tpose.position,
                         dtype=float).reshape(-1, 3)
        rows.append({"subject": f"p{m.group('subject')}",
                     "model_top_m": pos[:, 2].max()})
        del raw

    df = pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)
    heights = load_tablefeats(Path(data_dir) / "TableFeats.mat") \
        .groupby("sub")["height"].first()

    df["sub_id"] = df.subject.map(ds.SUBJECT_ID)
    df["height_tf"] = df.sub_id.map(heights)

    scale = (df.height_tf / (df.model_top_m * 100)).median()
    df["implied_cm"] = df.model_top_m * 100 * scale
    df["err_cm"] = (df.implied_cm - df.height_tf).abs()
    r = np.corrcoef(df.model_top_m, df.height_tf)[0, 1]

    print(df[["subject", "sub_id", "model_top_m", "height_tf",
              "implied_cm", "err_cm"]].to_string(index=False, float_format="%.3f"))

    rep.add(OK if r > 0.999 else FAIL, "stature reproduces TableFeats heights",
            f"r = {r:.4f}, max error {df.err_cm.max():.2f} cm")
    rep.add(OK if set(df.sub_id) == set(heights.index) else FAIL,
            "SUBJECT_ID covers every TableFeats subject",
            f"mapped {sorted(df.sub_id)} vs {sorted(heights.index)}")


def run(data_dir: Path | None = None, full: bool = False) -> int:
    data_dir = ds.find_data_dir(data_dir)
    rep = Report()
    audit_readme(data_dir, rep)
    if full:
        verify_subject_map(data_dir, rep)
    print(f"\n=== {rep.unexpected} unexpected, {rep.known} known defects, "
          f"{rep.warned} warnings ===")
    if rep.known:
        print("KNOWN entries are defects in the published dataset, not code "
              "bugs; dataset.py works around them. See NOTES.md.")
    if rep.unexpected:
        print("UNEXPECTED failures mean the data or the assumptions changed.")
    return rep.unexpected
