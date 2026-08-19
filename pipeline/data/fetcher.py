"""
Databento data fetcher with batch downloading, retry logic, local
caching, and free-trial budget tracking.

Each 2-trading-day batch is saved as a separate Parquet file immediately
on success, so interrupted runs can resume from the last completed batch.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import databento as db
import pandas as pd

from .config import PipelineConfig, TZ, US_HOLIDAYS_2025


class DatabentoFetcher:
    """Downloads NOII and L1 data from Databento in resumable batches."""

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.client = db.Historical(config.api_key)
        self._spent: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_all(self) -> None:
        """Download all batches to disk. Does not load into memory.

        Each batch is cached as a separate Parquet file. Re-runs skip
        already-downloaded batches automatically.
        """
        batches = self._make_batches()
        print(f"Trading days: {sum(len(b) for b in batches)}, "
              f"batches: {len(batches)}")

        self.cfg.noii_raw_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.l1_raw_dir.mkdir(parents=True, exist_ok=True)

        for i, batch_dates in enumerate(batches):
            label = self._batch_label(batch_dates)
            noii_path = self.cfg.noii_raw_dir / f"{label}.parquet"
            l1_path = self.cfg.l1_raw_dir / f"{label}.parquet"

            if noii_path.exists() and l1_path.exists():
                print(f"  [{i+1}/{len(batches)}] {label}  cached, skipping")
                continue

            self._budget_gate(batch_dates, interactive=(i > 0))

            print(f"  [{i+1}/{len(batches)}] {label}  downloading ...")
            start_dt, end_dt = self._batch_window(batch_dates)

            if not noii_path.exists():
                noii_df = self._download_with_retry(
                    schema=self.cfg.noii_schema,
                    start=start_dt,
                    end=end_dt,
                    label=f"NOII {label}",
                )
                if noii_df is not None:
                    noii_df.to_parquet(noii_path)

            if not l1_path.exists():
                l1_parts: list[pd.DataFrame] = []
                for d in batch_dates:
                    day_start = dt.datetime.combine(
                        d, dt.time(15, 49, 0), tzinfo=TZ
                    )
                    day_end = dt.datetime.combine(
                        d, dt.time(16, 0, 30), tzinfo=TZ
                    )
                    part = self._download_with_retry(
                        schema=self.cfg.l1_schema,
                        start=day_start,
                        end=day_end,
                        label=f"L1 {d.isoformat()}",
                    )
                    if part is not None and not part.empty:
                        l1_parts.append(part)
                l1_df = (
                    pd.concat(l1_parts, ignore_index=True)
                    if l1_parts
                    else pd.DataFrame()
                )
                if not l1_df.empty:
                    print(f"    L1 combined: {len(l1_df):,} rows")
                l1_df.to_parquet(l1_path)

    # ------------------------------------------------------------------
    # Batch construction
    # ------------------------------------------------------------------

    def _trading_dates(self) -> list[dt.date]:
        """Business days in the configured range, excluding US holidays."""
        bdays = pd.bdate_range(
            self.cfg.start_date, self.cfg.end_date, freq="B"
        )
        holidays = set(US_HOLIDAYS_2025)
        return [d.date() for d in bdays if d.date() not in holidays]

    def _make_batches(self) -> list[list[dt.date]]:
        """Split trading dates into groups of batch_size_days."""
        dates = self._trading_dates()
        n = self.cfg.batch_size_days
        return [dates[i:i + n] for i in range(0, len(dates), n)]

    @staticmethod
    def _batch_label(dates: list[dt.date]) -> str:
        if len(dates) == 1:
            return dates[0].isoformat()
        return f"{dates[0].isoformat()}_to_{dates[-1].isoformat()}"

    def _batch_window(
        self, dates: list[dt.date]
    ) -> tuple[dt.datetime, dt.datetime]:
        """Datetime range that covers the auction window for all dates
        in this batch.

        Start: first date at 15:49:00 (one minute early buffer).
        End:   last date at 16:00:30 (half-second buffer past close).
        """
        first = dt.datetime.combine(
            dates[0], dt.time(15, 49, 0), tzinfo=TZ
        )
        last = dt.datetime.combine(
            dates[-1], dt.time(16, 0, 30), tzinfo=TZ
        )
        return first, last

    # ------------------------------------------------------------------
    # Download with retry
    # ------------------------------------------------------------------

    def _download_with_retry(
        self,
        schema: str,
        start: dt.datetime,
        end: dt.datetime,
        label: str,
    ) -> pd.DataFrame | None:
        """Single download attempt with exponential-backoff retry.

        Returns a DataFrame on success, or None after all retries fail.
        """
        delay = self.cfg.initial_retry_delay_sec

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                data = self.client.timeseries.get_range(
                    dataset=self.cfg.dataset,
                    symbols=self.cfg.symbols,
                    schema=schema,
                    start=start,
                    end=end,
                )
                df = data.to_df(tz=TZ)
                print(f"    {label}: {len(df):,} rows")
                return df

            except Exception as exc:
                if attempt < self.cfg.max_retries:
                    print(f"    {label}: attempt {attempt} failed "
                          f"({exc}), retrying in {delay:.0f}s ...")
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
                else:
                    print(f"    {label}: all {self.cfg.max_retries} "
                          f"attempts failed. Last error: {exc}")
                    print(f"    Stopping. Previously downloaded batches "
                          f"are saved in {self.cfg.raw_dir}")
                    return None

    # ------------------------------------------------------------------
    # Budget tracking
    # ------------------------------------------------------------------

    def _estimate_batch_cost(self, dates: list[dt.date]) -> float:
        """Ask Databento for the cost of one batch (NOII + L1)."""
        start_dt, end_dt = self._batch_window(dates)
        total = 0.0
        for schema in (self.cfg.noii_schema, self.cfg.l1_schema):
            try:
                cost = self.client.metadata.get_cost(
                    dataset=self.cfg.dataset,
                    symbols=self.cfg.symbols,
                    schema=schema,
                    start=start_dt,
                    end=end_dt,
                )
                total += cost
            except Exception:
                pass
        return total

    def _budget_gate(
        self, batch_dates: list[dt.date], interactive: bool = True
    ) -> None:
        """Check remaining budget. Only prompt the user when balance
        drops below the warning threshold."""
        try:
            batch_cost = self._estimate_batch_cost(batch_dates)
        except Exception:
            return

        self._spent += batch_cost
        remaining = self.cfg.initial_budget - self._spent

        if remaining < self.cfg.budget_warn_threshold:
            label = self._batch_label(batch_dates)
            print(f"\n  ** Budget alert **")
            print(f"     Next batch ({label}): ~${batch_cost:.2f}")
            print(f"     Estimated spent so far: ~${self._spent:.2f}")
            print(f"     Remaining free trial: ~${remaining:.2f}")
            if interactive:
                ans = input("     Continue? [y/n]: ").strip().lower()
                if ans != "y":
                    print("     Stopped by user. All completed batches "
                          "are saved.")
                    raise SystemExit(0)

    # ------------------------------------------------------------------
    # Pre-save filter
    # ------------------------------------------------------------------

    def _filter_auction_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows within the auction window (15:49-16:00 ET).

        Applied to L1 data before saving to disk, since the batch time
        range can span a full day but we only need the last 11 minutes.
        """
        ts = df["ts_event"]
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(TZ)
        else:
            ts = ts.dt.tz_convert(TZ)
        hour_min = ts.dt.hour * 100 + ts.dt.minute
        return df[(hour_min >= 1549) & (hour_min <= 1600)].copy()

    # ------------------------------------------------------------------
    # Load cached data
    # ------------------------------------------------------------------

    def _load_cached(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Read all cached batch Parquets and concatenate."""
        noii_frames = self._read_dir(self.cfg.noii_raw_dir)
        l1_frames = self._read_dir(self.cfg.l1_raw_dir)
        print(f"Loaded from cache: NOII {len(noii_frames):,} rows, "
              f"L1 {len(l1_frames):,} rows")
        return noii_frames, l1_frames

    @staticmethod
    def _read_dir(directory: Path) -> pd.DataFrame:
        files = sorted(directory.glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        parts = [pd.read_parquet(f) for f in files]
        return pd.concat(parts, ignore_index=True)
