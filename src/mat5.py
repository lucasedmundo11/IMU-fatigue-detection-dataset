"""Minimal MAT-file level-5 element reader.

scipy handles ordinary .mat variables fine, but it will not descend into the
MCOS subsystem blob that holds a MATLAB `table`'s payload. That blob is a plain
MAT5 element stream, so we walk it directly.

Only what `tablefeats` needs is implemented: tag decoding and a recursive
harvester. Full struct/cell reconstruction is deliberately absent -- the MCOS
`FileWrapper__` wrapper does not follow the documented struct layout, so
harvesting numeric blocks and aligning them to the recovered column names is
both simpler and more robust than modelling the object graph.

Reference: MAT-File Format, "Level 5 MAT-File Format" (MathWorks).
"""

from __future__ import annotations

import struct

import numpy as np

miMATRIX = 14

# MAT5 data type -> numpy dtype string
NUMPY_OF = {
    1: "i1",    # miINT8
    2: "u1",    # miUINT8
    3: "i2",    # miINT16
    4: "u2",    # miUINT16
    5: "i4",    # miINT32
    6: "u4",    # miUINT32
    7: "f4",    # miSINGLE
    9: "f8",    # miDOUBLE
    12: "i8",   # miINT64
    13: "u8",   # miUINT64
    16: "u1",   # miUTF8
    17: "u2",   # miUTF16
    18: "u4",   # miUTF32
}


def read_tag(buf: bytes, pos: int) -> tuple[int, int, int, int]:
    """Decode one element tag.

    Returns (dtype, nbytes, data_offset, next_pos). Handles both the 8-byte
    tag and the compact form, where a non-zero upper half-word carries the
    byte count and the data is packed into the same 8 bytes.
    """
    (w0,) = struct.unpack_from("<I", buf, pos)
    packed = (w0 >> 16) & 0xFFFF
    if packed:
        return w0 & 0xFFFF, packed, pos + 4, pos + 8
    (nbytes,) = struct.unpack_from("<I", buf, pos + 4)
    pad = (8 - nbytes % 8) % 8
    return w0, nbytes, pos + 8, pos + 8 + nbytes + pad


def harvest(buf: bytes, pos: int, end: int, depth: int = 0,
            out: list | None = None, max_depth: int = 40) -> list:
    """Collect every numeric element in [pos, end), recursing into miMATRIX.

    Returns a list of (depth, offset, ndarray). Offset is the position of the
    element's data in `buf`, which is what fixes column order in a table.
    """
    if out is None:
        out = []
    if depth > max_depth:
        return out

    while pos < end - 8:
        try:
            dtype, nbytes, off, nxt = read_tag(buf, pos)
        except struct.error:
            break
        if nxt <= pos or nxt > end + 8:
            break

        if dtype == miMATRIX:
            harvest(buf, off, off + nbytes, depth + 1, out, max_depth)
        elif dtype in NUMPY_OF:
            np_t = np.dtype(NUMPY_OF[dtype])
            count = nbytes // np_t.itemsize
            if count:
                out.append((depth, off, np.frombuffer(
                    buf, dtype=np_t.newbyteorder("<"), count=count, offset=off)))
        pos = nxt

    return out
