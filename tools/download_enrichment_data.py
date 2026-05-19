"""
Enrichment Dataset Downloader
Downloads supplementary datasets recommended in doc/202604/16/dataset_enrichment.md

Phases:
  1  VIX daily series (FRED: VIXCLS) → cache/vix.csv
     FOMC meeting dates (hardcoded 2015-2026) → cache/fomc_dates.csv
  2  Earnings dates per S&P 500 ticker (Nasdaq earnings calendar API)
     → cache/earnings_dates.csv
     Strategy: loop every business day 2015-present, record which S&P 500
     tickers had earnings that day. ~2900 requests, ~25 min.

Note: Sector ETF download (doc Phase 3) was removed — Stooq now requires
API keys for ETF tickers, and Yahoo Finance / yfinance is rate-limited.

Usage:
    python tools/download_enrichment_data.py --phase 1
    python tools/download_enrichment_data.py --phase 2
    python tools/download_enrichment_data.py --all
"""

import argparse
import io
import os
import sys
import time

import pandas as pd
import requests



# ---------------------------------------------------------------------------
# FOMC decision dates (announcement day of each meeting), 2015-2026
# Source: Federal Reserve (https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)
# ---------------------------------------------------------------------------
FOMC_DATES = [
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17",
    "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15",
    "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
    "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020 (includes two emergency cuts in March)
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (scheduled)
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_sp500_tickers() -> list:
    path = "cache/sp500_list.csv"
    if not os.path.exists(path):
        print(f"ERROR: {path} not found — run update_sp500.py first")
        sys.exit(1)
    df = pd.read_csv(path)
    return df["Symbol"].dropna().str.strip().tolist()


# ---------------------------------------------------------------------------
# Phase 1: VIX + FOMC
# ---------------------------------------------------------------------------

def phase1_vix():
    """
    Download VIX from FRED (Federal Reserve public CSV — no auth required).
    Series: VIXCLS  https://fred.stlouisfed.org/series/VIXCLS
    """
    print("\n--- VIX Download (FRED: VIXCLS) ---")
    out = "cache/vix.csv"
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
        df.columns = ["Date", "VIX"]
        # FRED uses "." for non-trading days — drop those rows
        df = df[df["VIX"] != "."].copy()
        df["VIX"] = df["VIX"].astype(float)
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        df = df.sort_values("Date").reset_index(drop=True)
        df.to_csv(out, index=False)
        print(f"  Saved {len(df)} rows → {out}")
    except Exception as e:
        print(f"  FAILED: {e}")


def phase1_fomc():
    print("\n--- FOMC Dates (hardcoded 2015-2026) ---")
    out = "cache/fomc_dates.csv"
    df = pd.DataFrame({"Date": FOMC_DATES})
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df.to_csv(out, index=False)
    print(f"  Saved {len(df)} rows → {out}")


# ---------------------------------------------------------------------------
# Phase 2: Earnings dates via Nasdaq earnings calendar API
# ---------------------------------------------------------------------------

NASDAQ_EARNINGS_URL = "https://api.nasdaq.com/api/calendar/earnings"
NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
}


def _fetch_nasdaq_earnings_for_date(date_str: str) -> list[str]:
    """
    Call Nasdaq earnings calendar API for one date.
    Returns list of ticker symbols that reported earnings on that date.
    """
    try:
        r = requests.get(
            NASDAQ_EARNINGS_URL,
            params={"date": date_str},
            headers=NASDAQ_HEADERS,
            timeout=15,
        )
        if r.status_code == 429:
            return None  # caller handles backoff
        r.raise_for_status()
        data = r.json()
        rows = data.get("data", {}).get("rows") or []
        return [row["symbol"].upper() for row in rows if row.get("symbol")]
    except Exception:
        return []


def phase2_earnings():
    """
    Build earnings date database by querying the Nasdaq earnings calendar
    for every business day from 2015-01-01 to today. No auth required.
    Output: cache/earnings_dates.csv with columns [Ticker, Date]
    (one row per earnings event — a ticker may appear multiple times across years).
    """
    print("\n--- Earnings Dates (Nasdaq calendar API, 2015-present) ---")
    out = "cache/earnings_dates.csv"
    checkpoint_file = "cache/earnings_dates_checkpoint.txt"

    sp500 = set(_load_sp500_tickers())

    # Generate all US business days from 2015-01-01 to today+90 days
    # (Nasdaq API has earnings estimates ~2 months out)
    end = pd.Timestamp.today() + pd.Timedelta(days=90)
    all_dates = pd.bdate_range("2015-01-01", end).strftime("%Y-%m-%d").tolist()

    # Resume: load last processed date from checkpoint
    start_from = all_dates[0]
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            start_from = f.read().strip()
        print(f"  Resuming from {start_from}")

    remaining_dates = [d for d in all_dates if d >= start_from]
    print(f"  {len(remaining_dates)} business days to process (~{len(remaining_dates)*0.4/60:.0f} min)")

    # Load existing records
    records = []
    if os.path.exists(out):
        records = pd.read_csv(out).to_dict("records")
        print(f"  {len(records)} existing records loaded")

    consecutive_429 = 0

    for idx, date_str in enumerate(remaining_dates):
        tickers = _fetch_nasdaq_earnings_for_date(date_str)

        if tickers is None:  # rate limited
            consecutive_429 += 1
            wait = 30 * (2 ** min(consecutive_429, 4))
            print(f"\n  [429 on {date_str} — waiting {wait}s]", flush=True)
            time.sleep(wait)
            tickers = _fetch_nasdaq_earnings_for_date(date_str) or []
        else:
            consecutive_429 = 0

        # Keep only S&P 500 tickers
        matched = [t for t in tickers if t in sp500]
        for ticker in matched:
            records.append({"Ticker": ticker, "Date": date_str})

        if matched:
            print(f"  {date_str}: {', '.join(matched)}")

        # Checkpoint every 100 days
        if (idx + 1) % 100 == 0:
            pd.DataFrame(records).to_csv(out, index=False)
            with open(checkpoint_file, "w") as f:
                f.write(date_str)
            pct = (idx + 1) / len(remaining_dates) * 100
            print(f"  [checkpoint @ {date_str} — {pct:.0f}% done, {len(records)} records]")

        time.sleep(0.4)

    df = pd.DataFrame(records).drop_duplicates().sort_values(["Ticker", "Date"])
    df.to_csv(out, index=False)
    # Clean up checkpoint file on successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    print(f"\n  Saved {len(df)} earnings events for {df['Ticker'].nunique()} tickers → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download enrichment datasets")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--phase", type=int, choices=[1, 2], help="Run a specific phase")
    group.add_argument("--all", action="store_true", help="Run all phases sequentially")
    args = parser.parse_args()

    os.makedirs("cache", exist_ok=True)

    if args.all or args.phase == 1:
        print("=" * 60)
        print("PHASE 1: VIX + FOMC Dates")
        print("=" * 60)
        phase1_vix()
        phase1_fomc()

    if args.all or args.phase == 2:
        print("=" * 60)
        print("PHASE 2: Earnings Dates")
        print("=" * 60)
        phase2_earnings()

    print("\nDone.")


if __name__ == "__main__":
    main()
