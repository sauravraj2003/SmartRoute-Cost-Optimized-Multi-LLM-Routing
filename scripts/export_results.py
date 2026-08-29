"""Convert benchmark results between CSV and Parquet."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="benchmark/results/matrix.csv")
    parser.add_argument("--output", default="benchmark/results/matrix.parquet")
    args = parser.parse_args()
    src, dst = Path(args.input), Path(args.output)
    frame = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix == ".parquet":
        frame.to_parquet(dst, index=False)
    else:
        frame.to_csv(dst, index=False)
    print(f"Exported {len(frame)} rows to {dst}")


if __name__ == "__main__":
    main()
