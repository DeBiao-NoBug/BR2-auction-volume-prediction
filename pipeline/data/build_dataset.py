#!/usr/bin/env python3
"""
Entry point: fetch Databento data and build an Optiver-format Parquet.

Usage:
    python -m pipeline.data.build_dataset          # full run
    python -m pipeline.data.build_dataset --dry-run # cost estimate only
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .config import PipelineConfig
from .fetcher import DatabentoFetcher
from .processor import DataProcessor


def main(dry_run: bool = False) -> None:
    config = PipelineConfig()

    print("=" * 60)
    print("Databento -> Optiver-format dataset builder")
    print("=" * 60)
    print(f"Symbols:    {len(config.symbols)} stocks")
    print(f"Date range: {config.start_date} to {config.end_date}")
    print(f"Schemas:    NOII={config.noii_schema}, L1={config.l1_schema}")
    print(f"Output:     {config.output_path}")
    print()

    fetcher = DatabentoFetcher(config)

    if dry_run:
        _dry_run(fetcher)
        return

    # Phase 1: download (with caching and retry)
    print("Phase 1: Downloading raw data")
    print("-" * 40)
    fetcher.download_all()

    # Phase 2: process batch by batch to avoid loading all L1 into memory
    print()
    print("Phase 2: Processing batch by batch")
    print("-" * 40)
    processor = DataProcessor(config)
    batches = fetcher._make_batches()
    processed = []

    for i, batch_dates in enumerate(batches):
        label = fetcher._batch_label(batch_dates)
        noii_path = config.noii_raw_dir / f"{label}.parquet"
        l1_path = config.l1_raw_dir / f"{label}.parquet"

        if not noii_path.exists():
            print(f"  [{i+1}/{len(batches)}] {label}  no NOII, skipping")
            continue

        noii = pd.read_parquet(noii_path)
        l1 = pd.read_parquet(l1_path) if l1_path.exists() else pd.DataFrame()

        if noii.empty:
            continue

        batch_df = processor.merge_batch(noii, l1)
        processed.append(batch_df)
        del noii, l1

        print(f"  [{i+1}/{len(batches)}] {label}  {len(batch_df):,} rows")

    if not processed:
        print("No data to process. Exiting.")
        sys.exit(1)

    # Phase 3: finalize (assign IDs, add sector, format columns)
    print()
    print("Phase 3: Finalizing")
    print("-" * 40)
    merged = pd.concat(processed, ignore_index=True)
    del processed
    dataset = processor.finalize(merged)

    # Phase 4: save
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(config.output_path, index=False)
    print(f"\nSaved to {config.output_path}")
    print(f"File size: {config.output_path.stat().st_size / 1e6:.1f} MB")


def _dry_run(fetcher: DatabentoFetcher) -> None:
    """Estimate total cost without downloading anything."""
    batches = fetcher._make_batches()
    total = 0.0
    for i, batch_dates in enumerate(batches):
        label = fetcher._batch_label(batch_dates)
        cost = fetcher._estimate_batch_cost(batch_dates)
        total += cost
        if i < 3 or i == len(batches) - 1:
            print(f"  Batch {i+1}: {label}  ~${cost:.4f}")
        elif i == 3:
            print(f"  ... ({len(batches) - 4} more batches) ...")

    print(f"\nEstimated total: ${total:.2f}")
    remaining = fetcher.cfg.initial_budget - total
    print(f"Free trial after download: ~${remaining:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Optiver-format dataset from Databento"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate cost without downloading",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
