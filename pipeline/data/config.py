"""
Configuration for the Databento data pipeline.

Symbols, date ranges, paths, schema constants, and the NASDAQ 100
constituent list used as the default stock universe.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/New_York")

# NASDAQ 100 constituents as of January 2025.
# Update this list if composition changes during the sample period.
NASDAQ_100: list[str] = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP",
    "ADSK", "AEP", "AMAT", "AMD", "AMGN",
    "AMZN", "ANSS", "APP", "ARM", "ASML",
    "AVGO", "AZN", "BIIB", "BKNG", "BKR",
    "CCEP", "CDNS", "CDW", "CEG", "CHTR",
    "CMCSA", "COIN", "COST", "CPRT", "CRWD",
    "CSGP", "CTAS", "CTSH", "DASH", "DDOG",
    "DLTR", "DXCM", "EA", "EXC", "FANG",
    "FAST", "FTNT", "GEHC", "GFS", "GILD",
    "GOOG", "GOOGL", "HON", "IDXX", "INTC",
    "INTU", "ISRG", "KDP", "KHC", "KLAC",
    "LIN", "LRCX", "LULU", "MAR", "MCHP",
    "MDB", "MDLZ", "MELI", "META", "MNST",
    "MRNA", "MRVL", "MSFT", "MU", "NFLX",
    "NVDA", "NXPI", "ODFL", "ON", "ORLY",
    "PANW", "PAYX", "PCAR", "PDD", "PEP",
    "PLTR", "PYPL", "QCOM", "REGN", "ROP",
    "ROST", "SBUX", "SMCI", "SNPS", "TEAM",
    "TMUS", "TSLA", "TTD", "TTWO", "TXN",
    "VRSK", "VRTX", "WBD", "WDAY", "XEL",
]

US_HOLIDAYS_2025: list[dt.date] = [
    dt.date(2025, 1, 20),   # MLK Day
    dt.date(2025, 2, 17),   # Presidents' Day
    dt.date(2025, 4, 18),   # Good Friday
    dt.date(2025, 5, 26),   # Memorial Day
    dt.date(2025, 6, 19),   # Juneteenth
    dt.date(2025, 7, 4),    # Independence Day
    dt.date(2025, 9, 1),    # Labor Day
    dt.date(2025, 11, 27),  # Thanksgiving
    dt.date(2025, 12, 25),  # Christmas
]

@dataclass
class PipelineConfig:
    """All tunables for the fetch-and-build pipeline."""

    # Databento credentials (set DATABENTO_API_KEY in your environment)
    api_key: str = field(
        default_factory=lambda: os.environ.get("DATABENTO_API_KEY", "")
    )

    # Databento dataset and schemas
    dataset: str = "XNAS.ITCH"
    noii_schema: str = "imbalance"
    l1_schema: str = "mbp-1"

    # Stock universe
    symbols: list[str] = field(default_factory=lambda: NASDAQ_100)

    # Date range (inclusive on both ends)
    start_date: dt.date = dt.date(2025, 1, 6)
    end_date: dt.date = dt.date(2025, 12, 31)

    # Auction window (Eastern Time)
    auction_start: dt.time = dt.time(15, 50, 0)
    auction_end: dt.time = dt.time(16, 0, 0)
    bucket_interval_sec: int = 10

    # Download settings
    batch_size_days: int = 2
    max_retries: int = 5
    initial_retry_delay_sec: float = 5.0

    # Budget tracking (free trial threshold)
    initial_budget: float = 120.0
    budget_warn_threshold: float = 20.0

    # Paths (relative to this file's grandparent = Our BR_2/)
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )

    @property
    def raw_dir(self) -> Path:
        return self.project_root / "pipeline" / "data" / "raw"

    @property
    def noii_raw_dir(self) -> Path:
        return self.raw_dir / "noii"

    @property
    def l1_raw_dir(self) -> Path:
        return self.raw_dir / "l1"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "pipeline" / "data" / "output"

    @property
    def output_path(self) -> Path:
        return self.output_dir / "dataset.parquet"
