"""Command line for the data-processing layer.

    python3 -m src inspect                 list recordings, channels, strides
    python3 -m src validate [--full]       check the data against the README
    python3 -m src export-tablefeats       decode TableFeats.mat to csv/parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import dataset as ds
from .tablefeats import load_tablefeats
from .validate import run as run_validation


def cmd_inspect(args) -> int:
    data_dir = ds.find_data_dir(args.data)
    recordings = ds.recording_paths(data_dir)
    strides = ds.strides_paths(data_dir)

    print(f"{len(recordings)} recordings, {len(strides)} stride files, "
          f"fs = {ds.FS:g} Hz\n")
    print(f"{'file':<32} {'sub':>4} {'run':<17} {'dir':<4} {'cond':<6} "
          f"{'samples':>10} {'dur_s':>8} {'laps':>7}")
    print("-" * 96)

    for path in recordings:
        rec = ds.load_recording(path)
        laps = ds.RUNS[rec.run][2]
        print(f"{path.stem:<32} {rec.sub_id:>4} {rec.run:<17} "
              f"{rec.direction or '-':<4} {rec.condition:<6} "
              f"{rec.n_samples:>10,} {rec.duration_s:>8.1f} "
              f"{f'{laps[0]}-{laps[-1]}':>7}")
        del rec

    print(f"\nchannels per recording: "
          f"{len(ds.SEGMENTS)} segments x (acc + angvel) x 3 axes "
          f"+ {len(ds.JOINTS)} joints x angle x 3 axes = "
          f"{len(ds.SEGMENTS) * 6 + len(ds.JOINTS) * 3}")
    print(f"   segment.acc     m/s^2")
    print(f"   segment.angvel  rad/s")
    print(f"   joint.angle     DEGREES (README says radians and is wrong)")

    s = ds.load_strides(strides[0])
    print(f"\nstride files: {len(s.channels)} channels x {s.n_points} "
          f"normalised points x n_strides")
    print(f"   joints  {', '.join(ds.STRIDE_JOINTS)}  (X / Y / resultant)")
    print(f"   bodies  {', '.join(ds.STRIDE_BODIES)}  (nacc / jerk / angvel)")
    for path in strides:
        st = ds.load_strides(path)
        print(f"   {st.subject} (sub {st.sub_id:>2}): {st.n_strides:>5} strides")
        del st
    return 0


def cmd_validate(args) -> int:
    return run_validation(ds.find_data_dir(args.data), full=args.full)


def cmd_export_tablefeats(args) -> int:
    df = load_tablefeats(ds.find_data_dir(args.data) / "TableFeats.mat")
    print(f"TableFeats: {df.shape[0]:,} rows x {df.shape[1]} numeric columns")

    import pandas as pd
    print("\nrows per subject x fatigue class:")
    print(pd.crosstab(df["sub"], df["fatigue"]))
    print("\nRPE by fatigue class:")
    print(df.groupby("fatigue")["rpe"].agg(["min", "mean", "max", "count"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        try:
            df.to_parquet(out)
        except Exception as exc:
            out = out.with_suffix(".csv")
            print(f"\nparquet unavailable ({type(exc).__name__}); "
                  f"falling back to csv")
            df.to_csv(out, index=False)
    else:
        df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python3 -m src", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="list recordings, channels and strides")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("validate", help="check the data against the README")
    p.add_argument("--full", action="store_true",
                   help="also re-derive the pXXX -> sub mapping from stature")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("export-tablefeats",
                       help="decode TableFeats.mat to a table file")
    p.add_argument("--out",
                   default=str(ds.PREPROCESSED / "TableFeats.csv"))
    p.set_defaults(func=cmd_export_tablefeats)

    for p in sub.choices.values():
        p.add_argument("--data", default=None,
                       help="dataset directory (default: auto-detect "
                            "under data/raw, or $FATIGUE_DATA)")

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
