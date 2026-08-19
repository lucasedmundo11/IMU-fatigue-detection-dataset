"""Data processing for the 4TU running-fatigue dataset (doi:10.4121/14307743.v1).

Reading and validation only -- feature extraction and modelling come later.

    from src import load_recording, load_strides, load_tablefeats

    for rec in iter_recordings():
        acc = rec.acc("PEL")            # (N, 3) m/s^2
        knee = rec.angle("RKNE")        # (N, 3) DEGREES, not radians
        laps = rec.lap_index()          # 1-13
        rec.paper_mask()                # laps 2-4 / 8-10 / 11-13
"""

from .dataset import (
    FS,
    JOINT_LABEL,
    JOINTS,
    RUNS,
    SEGMENT_LABEL,
    PREPROCESSED,
    SEGMENTS,
    STRIDE_BODIES,
    STRIDE_JOINTS,
    SUBJECT_ID,
    Recording,
    Strides,
    find_data_dir,
    iter_recordings,
    iter_strides,
    load_recording,
    load_strides,
    recording_paths,
    strides_paths,
    subjects,
)
from .tablefeats import load_tablefeats

__all__ = [
    "FS", "JOINTS", "JOINT_LABEL", "RUNS", "SEGMENTS",
    "PREPROCESSED", "SEGMENT_LABEL", "STRIDE_BODIES", "STRIDE_JOINTS", "SUBJECT_ID",
    "Recording", "Strides", "find_data_dir", "iter_recordings", "iter_strides",
    "load_recording", "load_strides", "load_tablefeats", "recording_paths",
    "strides_paths", "subjects",
]
