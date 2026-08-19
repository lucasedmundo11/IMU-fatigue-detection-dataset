"""Decode TableFeats.mat -- the authors' normalised ML feature table.

MATLAB stores a `table` as an MCOS classdef object, which scipy returns as an
opaque stub. The payload lives in the file's `__function_workspace__` blob, a
raw MAT5 element stream (see mat5.py).

Result: 12,513 rows (strides pooled over 8 subjects) x 169 declared columns.
165 are numeric doubles and are recovered here. Four are categorical char
arrays -- footstrike, dir, gender, d_l -- which are NOT recovered; they are
skipped so the remaining names line up with the numeric blocks in declaration
order.

Values are already z-scored per subject, as the dataset README states.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from .mat5 import harvest

# Stored as char arrays rather than doubles, so they carry no numeric block.
CATEGORICAL = ("footstrike", "dir", "gender", "d_l")

# Tokens after these belong to the table class definition, not the data.
NAME_STOP = ("CustomProps", "VariableCustomProps", "versionSavedFrom")

# Columns worth typing as integers once recovered.
INT_COLUMNS = ("sub", "fatigue", "rpe")


def column_names(raw: bytes) -> list[str]:
    """Recover the table's variable names, in declaration order.

    The names sit in one contiguous ASCII run inside the blob, starting at the
    first column (`rpe`) and ending where the table class metadata begins.
    """
    m = re.search(rb"rpe\x00", raw)
    if not m:
        raise ValueError("could not locate the varnames block in TableFeats")

    window = raw[m.start():m.start() + 0x4000]
    toks = [t.decode() for t in re.findall(rb"[A-Za-z][A-Za-z0-9_]{1,40}",
                                           window)]
    for stop in NAME_STOP:
        if stop in toks:
            return toks[:toks.index(stop)]
    return toks


def load_tablefeats(path: str | Path) -> pd.DataFrame:
    """Read TableFeats.mat into a DataFrame of its 165 numeric columns."""
    raw = sio.loadmat(path, squeeze_me=True,
                      struct_as_record=False)["__function_workspace__"].tobytes()

    numeric_names = [n for n in column_names(raw) if n not in CATEGORICAL]

    blocks = [(off, a) for _, off, a in harvest(raw, 8, len(raw))
              if a.dtype == np.float64]
    if not blocks:
        raise ValueError("no float64 blocks found in the MCOS subsystem")

    # Every table column shares the row count; the modal length is n_rows.
    n_rows = Counter(a.size for _, a in blocks).most_common(1)[0][0]
    columns = sorted((off, a) for off, a in blocks if a.size == n_rows)

    if len(columns) != len(numeric_names):
        raise ValueError(
            f"{len(columns)} numeric columns but {len(numeric_names)} names -- "
            "alignment would be unsafe"
        )

    df = pd.DataFrame({name: arr for name, (_, arr) in
                       zip(numeric_names, columns)})
    for col in INT_COLUMNS:
        if col in df:
            df[col] = df[col].astype("int64")
    return df
