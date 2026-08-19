"""
Data processor: merges NOII and L1 data, computes derived features,
constructs the forward-shifted target, and formats the output to match
the Optiver Split_Train_Data.csv column schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig, TZ

SECTOR_MAP_PATH = Path(__file__).resolve().parent / "sector_map.json"

EPS = 1e-9


class DataProcessor:
    """Transforms raw Databento downloads into an Optiver-format table."""

    def __init__(self, config: PipelineConfig):
        self.cfg = config

    def merge_batch(
        self, noii: pd.DataFrame, l1: pd.DataFrame
    ) -> pd.DataFrame:
        """Process a single batch: filter, merge, derive, build target.

        Safe to call on each 2-day batch independently because the
        target shift operates within stock-day groups.
        """
        if noii.empty:
            raise ValueError("NOII dataframe is empty, nothing to process")

        noii = self._filter_closing_auction(noii)
        noii = self._assign_date_and_bucket(noii)

        if not l1.empty:
            l1 = self._filter_auction_window_l1(l1)
            merged = self._merge_noii_l1(noii, l1)
        else:
            merged = noii

        df = self._compute_derived_columns(merged)
        df = self._construct_target(df)
        return df

    def finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign IDs, add sector, and select Optiver columns.

        Call this once on the concatenated result of all merge_batch
        outputs, so that stock_id and date_id mappings are globally
        consistent.
        """
        df = self._assign_ids(df)
        df = self._add_sector(df)
        df = self._to_optiver_schema(df)
        return df

    def run(self, noii: pd.DataFrame, l1: pd.DataFrame) -> pd.DataFrame:
        """Full pipeline on a single chunk (kept for small-scale tests)."""
        df = self.merge_batch(noii, l1)
        return self.finalize(df)

    # ------------------------------------------------------------------
    # Step 1: Filter NOII to closing auction only
    # ------------------------------------------------------------------

    def _filter_closing_auction(self, df: pd.DataFrame) -> pd.DataFrame:
        if "auction_type" in df.columns:
            df = df[df["auction_type"] == "C"].copy()
        return df

    # ------------------------------------------------------------------
    # Step 2: Parse timestamps into trading date and seconds_in_bucket
    # ------------------------------------------------------------------

    def _assign_date_and_bucket(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = df["ts_event"]
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(TZ)
        else:
            ts = ts.dt.tz_convert(TZ)

        df = df.copy()
        df["ts_event"] = ts
        df["trading_date"] = ts.dt.date

        # Seconds since 15:50:00 on the same day
        auction_open = pd.to_timedelta("15:50:00")
        time_of_day = (
            ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
        )
        auction_open_sec = 15 * 3600 + 50 * 60
        raw_offset = time_of_day - auction_open_sec

        # Snap to the nearest 10-second boundary
        bucket_sec = self.cfg.bucket_interval_sec
        df["seconds_in_bucket"] = (
            (raw_offset // bucket_sec) * bucket_sec
        ).astype(int)

        # Keep only rows within [0, 590] (the 10-minute window)
        df = df[
            (df["seconds_in_bucket"] >= 0)
            & (df["seconds_in_bucket"] <= 590)
        ].copy()

        # If multiple NOII updates land in the same bucket, keep the last
        df = (
            df.sort_values("ts_event")
            .groupby(["symbol", "trading_date", "seconds_in_bucket"])
            .last()
            .reset_index()
        )
        return df

    # ------------------------------------------------------------------
    # Step 3: Filter L1 to the auction window
    # ------------------------------------------------------------------

    def _filter_auction_window_l1(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = df["ts_event"]
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(TZ)
        else:
            ts = ts.dt.tz_convert(TZ)

        df = df.copy()
        df["ts_event"] = ts

        hour_min = ts.dt.hour * 100 + ts.dt.minute
        # Keep 15:49 through 16:00 (one minute buffer on each side)
        df = df[(hour_min >= 1549) & (hour_min <= 1600)].copy()
        return df

    # ------------------------------------------------------------------
    # Step 4: merge_asof NOII with L1
    # ------------------------------------------------------------------

    def _merge_noii_l1(
        self, noii: pd.DataFrame, l1: pd.DataFrame
    ) -> pd.DataFrame:
        # Standardize the L1 column names we need
        l1_cols = {
            "bid_px_00": "bid_price",
            "ask_px_00": "ask_price",
            "bid_sz_00": "bid_size",
            "ask_sz_00": "ask_size",
        }
        rename_map = {k: v for k, v in l1_cols.items() if k in l1.columns}
        l1_subset = l1.rename(columns=rename_map)

        keep = ["ts_event", "symbol"] + list(rename_map.values())
        keep = [c for c in keep if c in l1_subset.columns]
        l1_subset = l1_subset[keep].copy()
        l1_subset = l1_subset.sort_values("ts_event")

        noii_sorted = noii.sort_values("ts_event")

        merged = pd.merge_asof(
            noii_sorted,
            l1_subset,
            on="ts_event",
            by="symbol",
            direction="backward",
            tolerance=pd.Timedelta("30s"),
        )
        return merged

    # ------------------------------------------------------------------
    # Step 5: Compute derived columns
    # ------------------------------------------------------------------

    def _compute_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Core: rename NOII fields to Optiver names
        col_map = {
            "total_imbalance_qty": "imbalance_size",
            "paired_qty": "matched_size",
            "ref_price": "reference_price",
        }
        for src, dst in col_map.items():
            if src in df.columns:
                df[dst] = df[src].astype(float)

        # imbalance_buy_sell_flag: B (bid/buy) -> 1, A (ask/sell) -> -1, N -> 0
        if "side" in df.columns:
            mapping = {"B": 1, "A": -1}
            df["imbalance_buy_sell_flag"] = (
                df["side"].map(mapping).fillna(0).astype(int)
            )

        # Volume
        df["volume"] = df["imbalance_size"] + df["matched_size"]

        # WAP (if bid/ask data available)
        if {"bid_price", "ask_price", "bid_size", "ask_size"}.issubset(
            df.columns
        ):
            denom = df["bid_size"] + df["ask_size"]
            df["wap"] = np.where(
                denom > EPS,
                (df["bid_price"] * df["ask_size"]
                 + df["ask_price"] * df["bid_size"]) / denom,
                df["reference_price"],
            )
            df["bid_ask_spread"] = df["ask_price"] - df["bid_price"]
        else:
            df["wap"] = df["reference_price"]
            df["bid_price"] = np.nan
            df["ask_price"] = np.nan
            df["bid_size"] = np.nan
            df["ask_size"] = np.nan
            df["bid_ask_spread"] = np.nan

        # Near/far price (from Databento NOII fields, if present).
        # Databento uses 0.0 for unavailable prices; Optiver uses NaN.
        if "cont_book_clr_price" in df.columns:
            df["near_price"] = df["cont_book_clr_price"].replace(0.0, np.nan)
        else:
            df["near_price"] = np.nan

        if "auct_interest_clr_price" in df.columns:
            df["far_price"] = df["auct_interest_clr_price"].replace(0.0, np.nan)
        else:
            df["far_price"] = np.nan

        # Databento L1 sizes are uint32; cast to float64 for Optiver
        # compatibility and NaN support.
        for col in ("bid_size", "ask_size"):
            if col in df.columns:
                df[col] = df[col].astype(np.float64)

        return df

    # ------------------------------------------------------------------
    # Step 6: Construct the forward-shifted target
    # ------------------------------------------------------------------

    def _construct_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(
            ["symbol", "trading_date", "seconds_in_bucket"]
        ).copy()

        # target = volume 6 buckets (60 seconds) ahead, per stock per day
        shift_n = 60 // self.cfg.bucket_interval_sec  # 6
        df["target"] = (
            df.groupby(["symbol", "trading_date"])["volume"]
            .shift(-shift_n)
        )

        # Drop rows where target is NaN (last 6 buckets of each day)
        before = len(df)
        df = df.dropna(subset=["target"]).copy()
        print(f"Target construction: dropped {before - len(df):,} tail "
              f"rows (no forward target), {len(df):,} rows remaining")
        return df

    # ------------------------------------------------------------------
    # Step 7: Assign integer IDs and finalize column order
    # ------------------------------------------------------------------

    def _assign_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # stock_id: stable integer mapping from symbol (alphabetical)
        symbols_sorted = sorted(df["symbol"].unique())
        sym_to_id = {s: i for i, s in enumerate(symbols_sorted)}
        df["stock_id"] = df["symbol"].map(sym_to_id)

        # date_id: integer mapping from trading_date (chronological)
        dates_sorted = sorted(df["trading_date"].unique())
        date_to_id = {d: i for i, d in enumerate(dates_sorted)}
        df["date_id"] = df["trading_date"].map(date_to_id)

        # time_id: same as date_id (Optiver convention)
        df["time_id"] = df["date_id"]

        # row_id: Optiver format "{stock_id}_{date_id}_{seconds_in_bucket}"
        df["row_id"] = (
            df["stock_id"].astype(str) + "_"
            + df["date_id"].astype(str) + "_"
            + df["seconds_in_bucket"].astype(str)
        )

        return df

    def _add_sector(self, df: pd.DataFrame) -> pd.DataFrame:
        if not SECTOR_MAP_PATH.exists():
            print("Warning: sector_map.json not found, skipping sector column")
            return df

        with open(SECTOR_MAP_PATH) as f:
            sector_map = json.load(f)

        df = df.copy()
        df["sector"] = df["symbol"].map(sector_map)
        unmapped = df["sector"].isna().sum()
        if unmapped > 0:
            missing = df.loc[df["sector"].isna(), "symbol"].unique()
            print(f"Warning: {len(missing)} tickers missing from "
                  f"sector_map: {', '.join(missing)}")
        return df

    def _to_optiver_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and order columns to match Optiver Split_Train_Data."""
        optiver_cols = [
            "stock_id",
            "date_id",
            "seconds_in_bucket",
            "imbalance_size",
            "imbalance_buy_sell_flag",
            "reference_price",
            "matched_size",
            "far_price",
            "near_price",
            "bid_price",
            "bid_size",
            "ask_price",
            "ask_size",
            "wap",
            "target",
            "time_id",
            "row_id",
        ]

        # Keep extra columns that downstream code might use
        extra = ["symbol", "trading_date", "volume", "sector"]
        cols = optiver_cols + [c for c in extra if c in df.columns]
        cols = [c for c in cols if c in df.columns]

        df = df[cols].copy()
        df = df.sort_values(
            ["stock_id", "date_id", "seconds_in_bucket"]
        ).reset_index(drop=True)

        print(f"Final dataset: {len(df):,} rows, "
              f"{df['stock_id'].nunique()} stocks, "
              f"{df['date_id'].nunique()} trading days")
        return df
