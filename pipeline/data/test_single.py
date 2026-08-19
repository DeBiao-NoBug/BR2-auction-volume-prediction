#!/usr/bin/env python3
"""
Smoke test: fetch one stock for one day, run the processing pipeline,
and print the resulting dataset for manual inspection.

Bypasses the batch/cache logic in fetcher.py and calls the Databento
API directly, so no files are left behind in the production raw/ dir.

Usage:
    cd "Our BR_2"
    /opt/anaconda3/bin/python -m pipeline.data.test_single
"""

from __future__ import annotations

import datetime as dt

import databento as db
import pandas as pd

from .config import PipelineConfig, TZ
from .processor import DataProcessor


def main() -> None:
    cfg = PipelineConfig(
        symbols=["AAPL"],
        start_date=dt.date(2025, 1, 6),
        end_date=dt.date(2025, 1, 6),
    )
    client = db.Historical(cfg.api_key)

    start = dt.datetime.combine(
        cfg.start_date, dt.time(15, 49, 0), tzinfo=TZ
    )
    end = dt.datetime.combine(
        cfg.end_date, dt.time(16, 0, 30), tzinfo=TZ
    )

    print("=" * 60)
    print("Smoke test: AAPL, 2025-01-06")
    print("=" * 60)
    print(f"Window: {start} to {end}")
    print()

    # Fetch NOII
    print("Fetching NOII (imbalance) ...")
    noii_raw = client.timeseries.get_range(
        dataset=cfg.dataset,
        symbols=cfg.symbols,
        schema=cfg.noii_schema,
        start=start,
        end=end,
    )
    noii_df = noii_raw.to_df(tz=TZ)
    print(f"  NOII rows: {len(noii_df):,}")

    # Fetch L1
    print("Fetching L1 (mbp-1) ...")
    l1_raw = client.timeseries.get_range(
        dataset=cfg.dataset,
        symbols=cfg.symbols,
        schema=cfg.l1_schema,
        start=start,
        end=end,
    )
    l1_df = l1_raw.to_df(tz=TZ)
    print(f"  L1 rows:   {len(l1_df):,}")

    # Process through the same pipeline as production
    print("\nProcessing ...")
    processor = DataProcessor(cfg)
    dataset = processor.run(noii_df, l1_df)

    # Save
    out_dir = cfg.project_root / "pipeline" / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_single.parquet"
    dataset.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1e3:.1f} KB")

    # Inspect
    print("\n" + "=" * 60)
    print("Columns:")
    print("=" * 60)
    for i, col in enumerate(dataset.columns):
        print(f"  {i:2d}. {col:<30s} dtype={dataset[col].dtype}")

    print("\n" + "=" * 60)
    print(f"Shape: {dataset.shape}")
    print("=" * 60)
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", "{:.4f}".format,
    ):
        print(dataset.to_string(index=False))


if __name__ == "__main__":
    main()
