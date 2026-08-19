#!/usr/bin/env python3
"""
One-time script: fetch GICS sector for each ticker via yfinance
and save to sector_map.json.

Usage:
    /opt/anaconda3/bin/python -m pipeline.data.gen_sector_map
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yfinance as yf

from .config import NASDAQ_100

OUTPUT = Path(__file__).resolve().parent / "sector_map.json"


def main() -> None:
    sector_map: dict[str, str] = {}
    failed: list[str] = []

    for i, ticker in enumerate(NASDAQ_100):
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "")
            if sector:
                sector_map[ticker] = sector
                print(f"  [{i+1}/{len(NASDAQ_100)}] {ticker}: {sector}")
            else:
                failed.append(ticker)
                print(f"  [{i+1}/{len(NASDAQ_100)}] {ticker}: no sector found")
        except Exception as exc:
            failed.append(ticker)
            print(f"  [{i+1}/{len(NASDAQ_100)}] {ticker}: error ({exc})")
        time.sleep(0.3)

    OUTPUT.write_text(json.dumps(sector_map, indent=2, sort_keys=True))
    print(f"\nSaved {len(sector_map)} tickers to {OUTPUT}")

    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")
        print("You can manually add these to sector_map.json")


if __name__ == "__main__":
    main()
