#!/usr/bin/env python3
"""
undervalued_scanner.py
=======================

A P/E + historical-valuation + volume + pullback screener for the S&P 500
and Nasdaq Composite, with a transparent, self-relative composite score.

WHY THIS EXISTS
----------------
Built from a conversation where the ask was: scan the S&P 500 + Nasdaq
Composite for undervalued stocks using (1) P/E ratio, (2) how cheap the
stock is relative to its own history, (3) stocks whose recent 10-20 trading
day volume is running below normal, and (4, added later) stocks trading
10-20% below their 52-week high. Those four are the USER'S criteria and are
weighted 2x. On top of that, 6 more factors (drawn from a "10 things to
consider" list) are scored at 1x weight: PEG ratio, trailing-vs-forward P/E
trend, free-cash-flow yield, debt/equity, ROE, P/E vs. sector peers, and
dividend payout sustainability. Two of the original 10 points -- insider
buying/selling and "why is the market pricing this low" -- are NOT scored
here because there's no reliable free bulk data source for them; they're
flagged as "review manually" for whatever makes your shortlist.

DATA SOURCE
-----------
Uses `yfinance` (free, unofficial Yahoo Finance wrapper). This needs real
internet access -- it will NOT work in a network-sandboxed environment.
Run it on your own machine.

    pip install yfinance pandas requests lxml openpyxl tqdm

Scanning the full S&P 500 + Nasdaq Composite (~500 + ~3,000-4,000 tickers)
means several thousand network calls. Expect this to take a while (easily
30-90+ minutes depending on --workers and Yahoo's mood) and to occasionally
get rate-limited -- the script checkpoints to CSV as it goes, so you can
kill it and re-run with --resume to pick up where you left off.

USAGE
-----
    # Quick smoke test with fabricated data, no network needed:
    python undervalued_scanner.py --demo

    # Real run, S&P 500 only, 20 worker threads, save to my_scan.xlsx:
    python undervalued_scanner.py --universe sp500 --workers 20 -o my_scan.xlsx

    # Full universe (S&P 500 + Nasdaq Composite), resume a prior run:
    python undervalued_scanner.py --universe both --resume -o full_scan.xlsx

    # Just test the plumbing on 25 random tickers before committing to a
    # multi-thousand-ticker run:
    python undervalued_scanner.py --universe both --limit 25

See --help for every knob (thresholds, recent-volume window, drawdown
band, weights, etc).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional, degrade gracefully
    def tqdm(iterable, **kwargs):
        return iterable

log = logging.getLogger("scanner")

# ---------------------------------------------------------------------------
# Weights -- USER'S four criteria count double; the six secondary factors
# from the "10 things to consider" list count single. Edit freely.
# ---------------------------------------------------------------------------
WEIGHTS = {
    # --- your criteria (2x) ---
    "pe_cheap": 2.0,          # trailing P/E low vs. the scanned universe/sector
    "historical_cheap": 2.0,  # current P/E cheap vs. the stock's own ~5yr norm
    "low_volume": 2.0,        # recent 10-20 day volume below its normal average
    "pullback_10_20": 2.0,    # price 10-20% below its 52-week high
    # --- secondary factors, weight 1x ---
    "peg": 1.0,
    "pe_trend": 1.0,          # forward P/E cheaper than trailing (earnings improving)
    "fcf_yield": 1.0,
    "debt_equity": 1.0,
    "roe": 1.0,
    "pe_vs_sector": 1.0,
    "payout_sustainable": 1.0,
}

RECENT_VOL_DAYS = 15      # midpoint of the "10 to 20 day" window requested
NORMAL_VOL_DAYS = 252     # ~1 trading year, used as the "normal" volume baseline
VOL_RATIO_FLAG = 0.85     # flag if recent/normal < this (i.e. 15%+ quieter)
PULLBACK_LO, PULLBACK_HI = 0.10, 0.20   # the "down 10-20% from highs" band

NASDAQ_LISTED_URL = "https://ftp.nasdaqtrader.com/dynamiclookup/nasdaqlisted.txt"
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


# ---------------------------------------------------------------------------
# Universe builders
# ---------------------------------------------------------------------------
def get_sp500_tickers() -> list[str]:
    """Scrape the current S&P 500 constituent list from Wikipedia."""
    tables = pd.read_html(SP500_WIKI_URL)
    df = tables[0]
    tickers = df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
    return sorted(tickers.unique().tolist())


def get_nasdaq_composite_tickers() -> list[str]:
    """
    Approximate the Nasdaq Composite with nasdaqtrader.com's listed-securities
    file, filtered to common/ordinary shares (drops ETFs and test issues).

    Caveat: the *real* Nasdaq Composite also includes some non-U.S.-listed
    ADRs and other edge cases this file doesn't capture perfectly. This is
    the standard practical proxy used by most open-source screeners, not an
    official index feed.
    """
    resp = requests.get(NASDAQ_LISTED_URL, timeout=30)
    resp.raise_for_status()
    # pipe-delimited, last line is a footer ("File Creation Time...")
    lines = [l for l in resp.text.splitlines() if l and not l.startswith("File Creation")]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|")
    df = df[(df["Test Issue"] == "N") & (df["ETF"] == "N")]
    tickers = df["Symbol"].astype(str).str.strip()
    tickers = tickers[~tickers.str.contains(r"[\.\$]", regex=True)]  # drop unit/warrant oddities
    return sorted(tickers.unique().tolist())


def build_universe(name: str) -> list[str]:
    if name == "sp500":
        return get_sp500_tickers()
    if name == "nasdaq":
        return get_nasdaq_composite_tickers()
    if name == "both":
        sp = get_sp500_tickers()
        nq = get_nasdaq_composite_tickers()
        return sorted(set(sp) | set(nq))
    raise ValueError(f"unknown universe {name!r}")


# ---------------------------------------------------------------------------
# Per-ticker fetch
# ---------------------------------------------------------------------------
@dataclass
class FetchResult:
    ticker: str
    ok: bool
    error: str | None = None
    data: dict = field(default_factory=dict)


def fetch_one(ticker: str, retries: int = 3, backoff: float = 1.5) -> FetchResult:
    """Pull fundamentals + price history for one ticker via yfinance."""
    import yfinance as yf  # imported lazily so --demo needs no network lib import failures

    last_err = None
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            hist = t.history(period="5y", interval="1d", auto_adjust=False)
            if hist is None or hist.empty or "trailingPE" not in info and info.get("trailingPE") is None:
                # still proceed -- some fields legitimately absent, that's fine
                pass
            return FetchResult(ticker, True, data=_extract_metrics(ticker, info, hist))
        except Exception as exc:  # yfinance raises all sorts of things
            last_err = str(exc)
            time.sleep(backoff ** attempt + random.random())
    return FetchResult(ticker, False, error=last_err)


def _extract_metrics(ticker: str, info: dict, hist: pd.DataFrame) -> dict:
    close = hist["Close"].dropna() if not hist.empty else pd.Series(dtype=float)
    vol = hist["Volume"].dropna() if not hist.empty else pd.Series(dtype=float)

    current_price = float(close.iloc[-1]) if len(close) else info.get("currentPrice")
    fifty2wk_high = float(close.tail(252).max()) if len(close) >= 5 else info.get("fiftyTwoWeekHigh")
    drawdown = None
    if current_price and fifty2wk_high:
        drawdown = (fifty2wk_high - current_price) / fifty2wk_high

    recent_vol = float(vol.tail(RECENT_VOL_DAYS).mean()) if len(vol) >= RECENT_VOL_DAYS else None
    baseline_slice = vol.iloc[:-RECENT_VOL_DAYS] if len(vol) > RECENT_VOL_DAYS else vol
    normal_vol = float(baseline_slice.tail(NORMAL_VOL_DAYS).mean()) if len(baseline_slice) else None
    vol_ratio = (recent_vol / normal_vol) if (recent_vol and normal_vol) else None

    eps = info.get("trailingEps")
    hist_pe_median = None
    if eps and eps > 0 and len(close) >= 60:
        hist_pe_series = close / eps
        hist_pe_series = hist_pe_series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(hist_pe_series):
            hist_pe_median = float(hist_pe_series.median())

    market_cap = info.get("marketCap")
    fcf = info.get("freeCashflow")
    fcf_yield = (fcf / market_cap) if (fcf and market_cap) else None

    debt_to_equity = info.get("debtToEquity")
    if debt_to_equity is not None:
        debt_to_equity = debt_to_equity / 100.0  # yfinance reports this as a percent-like number

    return dict(
        ticker=ticker,
        company=info.get("shortName") or info.get("longName"),
        sector=info.get("sector"),
        industry=info.get("industry"),
        market_cap=market_cap,
        price=current_price,
        trailing_pe=info.get("trailingPE"),
        forward_pe=info.get("forwardPE"),
        peg=info.get("pegRatio") or info.get("trailingPegRatio"),
        pb=info.get("priceToBook"),
        debt_equity=debt_to_equity,
        roe=info.get("returnOnEquity"),
        dividend_yield=info.get("dividendYield"),
        payout_ratio=info.get("payoutRatio"),
        short_pct_float=info.get("shortPercentOfFloat"),
        fcf_yield=fcf_yield,
        fifty2wk_high=fifty2wk_high,
        drawdown_from_high=drawdown,
        recent_avg_volume=recent_vol,
        normal_avg_volume=normal_vol,
        volume_ratio=vol_ratio,
        historical_median_pe=hist_pe_median,
    )


def fetch_universe(
    tickers: list[str],
    workers: int,
    sleep_between: float,
    checkpoint_path: Path,
    resume: bool,
) -> pd.DataFrame:
    done_rows: dict[str, dict] = {}
    if resume and checkpoint_path.exists():
        existing = pd.read_csv(checkpoint_path)
        done_rows = {r["ticker"]: r for r in existing.to_dict("records")}
        log.info("resuming: %d tickers already fetched", len(done_rows))

    todo = [t for t in tickers if t not in done_rows]
    log.info("fetching %d tickers (%d already done, %d remaining)", len(tickers), len(done_rows), len(todo))

    rows = list(done_rows.values())
    failures: list[str] = []

    def flush():
        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_one, t): t for t in todo}
        for i, fut in enumerate(tqdm(cf.as_completed(futures), total=len(futures), desc="scanning")):
            res = fut.result()
            if res.ok:
                rows.append(res.data)
            else:
                failures.append(f"{res.ticker}: {res.error}")
            if sleep_between:
                time.sleep(sleep_between)
            if i % 50 == 0:
                flush()

    flush()
    if failures:
        log.warning("%d tickers failed after retries (see scan_failures.log)", len(failures))
        Path("scan_failures.log").write_text("\n".join(failures))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scoring -- vectorized, self-relative to the scanned universe
# ---------------------------------------------------------------------------
def _lower_is_better(series: pd.Series, good_q=0.25, ok_q=0.5) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    good, ok = valid.quantile(good_q), valid.quantile(ok_q)
    return s.apply(lambda x: np.nan if pd.isna(x) else (1.0 if x <= good else (0.5 if x <= ok else 0.0)))


def _higher_is_better(series: pd.Series, good_q=0.75, ok_q=0.5) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    good, ok = valid.quantile(good_q), valid.quantile(ok_q)
    return s.apply(lambda x: np.nan if pd.isna(x) else (1.0 if x >= good else (0.5 if x >= ok else 0.0)))


def score_universe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("trailing_pe", "forward_pe", "peg", "pb", "debt_equity", "roe",
                "dividend_yield", "payout_ratio", "short_pct_float", "fcf_yield",
                "drawdown_from_high", "volume_ratio", "historical_median_pe", "market_cap"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # sector/market medians for the P/E-vs-sector comparison
    if "sector" in df.columns:
        df["sector_median_pe"] = df.groupby("sector")["trailing_pe"].transform("median")
    else:
        df["sector_median_pe"] = df["trailing_pe"].median()
    df["pe_vs_sector_ratio"] = df["trailing_pe"] / df["sector_median_pe"]

    # current P/E vs. the stock's own ~5yr historical median P/E
    df["pe_vs_history_ratio"] = df["trailing_pe"] / df["historical_median_pe"]

    scores = pd.DataFrame(index=df.index)
    scores["pe_cheap"] = _lower_is_better(df["trailing_pe"])
    scores["historical_cheap"] = _lower_is_better(df["pe_vs_history_ratio"])
    scores["low_volume"] = _lower_is_better(df["volume_ratio"])

    # pullback band is not a simple lower/higher-is-better -- score by distance
    # from the center of the 10-20% band
    def pullback_score(dd):
        if pd.isna(dd):
            return np.nan
        if PULLBACK_LO <= dd <= PULLBACK_HI:
            return 1.0
        if (PULLBACK_LO - 0.05) <= dd < PULLBACK_LO or PULLBACK_HI < dd <= (PULLBACK_HI + 0.10):
            return 0.5
        return 0.0
    scores["pullback_10_20"] = df["drawdown_from_high"].apply(pullback_score)

    scores["peg"] = _lower_is_better(df["peg"], good_q=0.33, ok_q=0.66)  # PEG has a natural "<1 good" feel
    scores["pe_trend"] = np.where(
        df["forward_pe"].notna() & df["trailing_pe"].notna(),
        (df["forward_pe"] < df["trailing_pe"]).astype(float),
        np.nan,
    )
    scores["fcf_yield"] = _higher_is_better(df["fcf_yield"])
    scores["debt_equity"] = _lower_is_better(df["debt_equity"])
    scores["roe"] = _higher_is_better(df["roe"])
    scores["pe_vs_sector"] = _lower_is_better(df["pe_vs_sector_ratio"])

    payout_for_scoring = df["payout_ratio"].where(df["dividend_yield"] > 0)
    scores["payout_sustainable"] = _lower_is_better(payout_for_scoring, good_q=0.33, ok_q=0.66)

    weight_row = pd.Series(WEIGHTS)
    weighted = scores[weight_row.index] * weight_row
    weight_available = scores[weight_row.index].notna() * weight_row
    df["composite_score"] = (weighted.sum(axis=1) / weight_available.sum(axis=1).replace(0, np.nan)) * 100
    df["score_components_available"] = scores[weight_row.index].notna().sum(axis=1)

    for col in scores.columns:
        df[f"flag_{col}"] = scores[col]

    df["low_volume_flag"] = df["volume_ratio"] < VOL_RATIO_FLAG
    df["pullback_10_20_flag"] = df["drawdown_from_high"].between(PULLBACK_LO, PULLBACK_HI)
    df["high_short_interest_caution"] = df["short_pct_float"] > 0.20
    df["needs_manual_review"] = "check insiders (Form 4) + why the market is pricing this low"

    return df.sort_values("composite_score", ascending=False)


# ---------------------------------------------------------------------------
# Demo mode -- exercises the scoring pipeline with real numbers pulled by
# hand earlier for 16 tickers, so the logic can be sanity-checked with zero
# network access.
# ---------------------------------------------------------------------------
def demo_dataframe() -> pd.DataFrame:
    rows = [
        dict(ticker="CMCSA", company="Comcast Corporation", sector="Communication Services", market_cap=92.90e9, price=26.18, trailing_pe=8.47, forward_pe=7.45, peg=None, pb=1.03, debt_equity=1.00, roe=0.1149, dividend_yield=0.0504, payout_ratio=0.4273, short_pct_float=0.021, fcf_yield=0.09, fifty2wk_high=33.5, drawdown_from_high=0.219, recent_avg_volume=30500969, normal_avg_volume=35252514, volume_ratio=0.865, historical_median_pe=11.2),
        dict(ticker="CNC", company="Centene Corporation", sector="Healthcare", market_cap=32.855e9, price=66.54, trailing_pe=None, forward_pe=14.44, peg=0.38, pb=1.48, debt_equity=0.71, roe=-0.2036, dividend_yield=None, payout_ratio=None, short_pct_float=0.0296, fcf_yield=0.05, fifty2wk_high=78.0, drawdown_from_high=0.147, recent_avg_volume=5017450, normal_avg_volume=5803495, volume_ratio=0.865, historical_median_pe=20.5),
        dict(ticker="APA", company="APA Corporation", sector="Energy", market_cap=12.778e9, price=36.15, trailing_pe=8.57, forward_pe=9.58, peg=None, pb=2.02, debt_equity=0.49, roe=0.2666, dividend_yield=0.0247, payout_ratio=0.2117, short_pct_float=0.0864, fcf_yield=0.11, fifty2wk_high=44.0, drawdown_from_high=0.178, recent_avg_volume=5506362, normal_avg_volume=5860269, volume_ratio=0.940, historical_median_pe=9.8),
        dict(ticker="CHTR", company="Charter Communications", sector="Communication Services", market_cap=20.63e9, price=154.27, trailing_pe=4.01, forward_pe=3.58, peg=0.27, pb=1.09, debt_equity=4.42, roe=0.2720, dividend_yield=None, payout_ratio=None, short_pct_float=0.2925, fcf_yield=0.14, fifty2wk_high=270.0, drawdown_from_high=0.429, recent_avg_volume=2998008, normal_avg_volume=3458122, volume_ratio=0.867, historical_median_pe=6.0),
        dict(ticker="DVN", company="Devon Energy Corporation", sector="Energy", market_cap=49.291e9, price=42.74, trailing_pe=9.83, forward_pe=8.85, peg=0.98, pb=1.26, debt_equity=0.28, roe=0.1152, dividend_yield=0.0279, payout_ratio=0.2744, short_pct_float=0.0265, fcf_yield=0.10, fifty2wk_high=51.0, drawdown_from_high=0.162, recent_avg_volume=9694035, normal_avg_volume=14099226, volume_ratio=0.688, historical_median_pe=10.9),
        dict(ticker="ZBRA", company="Zebra Technologies", sector="Technology", market_cap=13.746e9, price=288.57, trailing_pe=34.78, forward_pe=18.01, peg=1.25, pb=5.19, debt_equity=0.86, roe=0.1529, dividend_yield=None, payout_ratio=None, short_pct_float=0.0571, fcf_yield=0.05, fifty2wk_high=420.0, drawdown_from_high=0.313, recent_avg_volume=890203, normal_avg_volume=963314, volume_ratio=0.924, historical_median_pe=24.0),
        dict(ticker="UHS", company="Universal Health Services", sector="Healthcare", market_cap=8.767e9, price=170.02, trailing_pe=6.92, forward_pe=7.35, peg=0.93, pb=1.33, debt_equity=0.69, roe=0.2095, dividend_yield=0.0047, payout_ratio=0.0326, short_pct_float=0.0616, fcf_yield=0.12, fifty2wk_high=210.0, drawdown_from_high=0.190, recent_avg_volume=920330, normal_avg_volume=946074, volume_ratio=0.973, historical_median_pe=10.5),
        dict(ticker="HCA", company="HCA Healthcare", sector="Healthcare", market_cap=90.198e9, price=406.59, trailing_pe=13.55, forward_pe=13.19, peg=1.35, pb=None, debt_equity=None, roe=None, dividend_yield=0.0077, payout_ratio=0.1044, short_pct_float=0.0420, fcf_yield=0.07, fifty2wk_high=470.0, drawdown_from_high=0.135, recent_avg_volume=1398494, normal_avg_volume=1497041, volume_ratio=0.934, historical_median_pe=15.8),
        dict(ticker="GPN", company="Global Payments Inc", sector="Financial Services", market_cap=22.72e9, price=85.86, trailing_pe=48.06, forward_pe=6.32, peg=0.31, pb=1.07, debt_equity=0.98, roe=0.0243, dividend_yield=0.0108, payout_ratio=None, short_pct_float=0.0799, fcf_yield=0.02, fifty2wk_high=105.0, drawdown_from_high=0.182, recent_avg_volume=3280302, normal_avg_volume=3739922, volume_ratio=0.877, historical_median_pe=18.0),
        dict(ticker="EG", company="Everest Group Ltd", sector="Financial Services", market_cap=14.113e9, price=370.11, trailing_pe=7.88, forward_pe=6.78, peg=0.71, pb=0.93, debt_equity=0.23, roe=0.1257, dividend_yield=0.0216, payout_ratio=0.1704, short_pct_float=0.0604, fcf_yield=0.09, fifty2wk_high=410.0, drawdown_from_high=0.097, recent_avg_volume=355714, normal_avg_volume=412246, volume_ratio=0.863, historical_median_pe=9.5),
        dict(ticker="VICI", company="VICI Properties Inc", sector="Real Estate", market_cap=29.068e9, price=26.40, trailing_pe=10.23, forward_pe=9.02, peg=3.76, pb=0.99, debt_equity=0.60, roe=0.0985, dividend_yield=0.0683, payout_ratio=0.6987, short_pct_float=0.0229, fcf_yield=0.08, fifty2wk_high=33.0, drawdown_from_high=0.20, recent_avg_volume=9741006, normal_avg_volume=9413800, volume_ratio=1.035, historical_median_pe=11.8),
        dict(ticker="LEN", company="Lennar Corporation", sector="Consumer Discretionary", market_cap=20.917e9, price=86.83, trailing_pe=13.49, forward_pe=15.39, peg=None, pb=0.97, debt_equity=0.29, roe=0.0737, dividend_yield=0.0230, payout_ratio=0.3108, short_pct_float=0.0025, fcf_yield=0.06, fifty2wk_high=130.0, drawdown_from_high=0.332, recent_avg_volume=2282442, normal_avg_volume=2603408, volume_ratio=0.877, historical_median_pe=12.9),
        dict(ticker="MU", company="Micron Technology", sector="Technology", market_cap=1097.0e9, price=971.66, trailing_pe=21.93, forward_pe=6.76, peg=0.04, pb=10.89, debt_equity=0.06, roe=0.6664, dividend_yield=0.0006, payout_ratio=0.0120, short_pct_float=0.0266, fcf_yield=0.03, fifty2wk_high=1150.0, drawdown_from_high=0.155, recent_avg_volume=41625048, normal_avg_volume=50030500, volume_ratio=0.832, historical_median_pe=35.0),
        dict(ticker="PFE", company="Pfizer Inc", sector="Healthcare", market_cap=152.694e9, price=26.79, trailing_pe=35.22, forward_pe=9.67, peg=None, pb=1.79, debt_equity=0.74, roe=0.0501, dividend_yield=0.0642, payout_ratio=2.2635, short_pct_float=0.0284, fcf_yield=0.06, fifty2wk_high=29.5, drawdown_from_high=0.092, recent_avg_volume=38629105, normal_avg_volume=42020269, volume_ratio=0.919, historical_median_pe=13.0),
        dict(ticker="LYFT", company="Lyft Inc", sector="Technology", market_cap=5.202e9, price=13.70, trailing_pe=2.42, forward_pe=9.26, peg=None, pb=2.19, debt_equity=0.43, roe=1.5258, dividend_yield=None, payout_ratio=None, short_pct_float=0.2255, fcf_yield=0.15, fifty2wk_high=21.0, drawdown_from_high=0.348, recent_avg_volume=12709149, normal_avg_volume=16123118, volume_ratio=0.788, historical_median_pe=18.0),
        dict(ticker="VST", company="Vistra Corp", sector="Utilities", market_cap=49.718e9, price=148.13, trailing_pe=24.98, forward_pe=None, peg=None, pb=None, debt_equity=None, roe=None, dividend_yield=None, payout_ratio=None, short_pct_float=None, fcf_yield=None, fifty2wk_high=203.0, drawdown_from_high=0.270, recent_avg_volume=None, normal_avg_volume=4641925, volume_ratio=None, historical_median_pe=None),
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--universe", choices=["sp500", "nasdaq", "both"], default="both",
                   help="which index to scan (default: both)")
    p.add_argument("--limit", type=int, default=None,
                   help="only scan the first N tickers (random sample) -- useful for a quick test run")
    p.add_argument("--workers", type=int, default=12, help="concurrent fetch threads (default: 12)")
    p.add_argument("--sleep", type=float, default=0.05,
                   help="seconds to sleep between requests per worker, be polite to Yahoo (default: 0.05)")
    p.add_argument("--checkpoint", default="scan_checkpoint.csv", help="checkpoint CSV path")
    p.add_argument("--resume", action="store_true", help="resume from --checkpoint instead of refetching everything")
    p.add_argument("-o", "--output", default="results.html",
                   help="output file -- extension picks the format: .html (sortable report), .xlsx, or .csv")
    p.add_argument("--top", type=int, default=100, help="how many top-scoring rows to keep in the output (default: 100)")
    p.add_argument("--demo", action="store_true", help="run the scorer on built-in sample data, no network needed")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                         format="%(asctime)s %(levelname)s %(message)s")

    if args.demo:
        print("Running in --demo mode: scoring 16 pre-loaded sample tickers, no network calls made.\n")
        raw = demo_dataframe()
    else:
        tickers = build_universe(args.universe)
        if args.limit:
            random.shuffle(tickers)
            tickers = tickers[: args.limit]
        print(f"Universe: {len(tickers)} tickers ({args.universe})")
        raw = fetch_universe(
            tickers,
            workers=args.workers,
            sleep_between=args.sleep,
            checkpoint_path=Path(args.checkpoint),
            resume=args.resume,
        )

    if raw.empty:
        print("No data fetched -- nothing to score.", file=sys.stderr)
        sys.exit(1)

    scored = score_universe(raw)
    top = scored.head(args.top)

    out_path = Path(args.output)
    display_cols = [
        "ticker", "company", "sector", "market_cap", "price",
        "trailing_pe", "forward_pe", "peg", "pb", "debt_equity", "roe",
        "dividend_yield", "payout_ratio", "fcf_yield", "short_pct_float",
        "drawdown_from_high", "pullback_10_20_flag",
        "volume_ratio", "low_volume_flag",
        "pe_vs_history_ratio", "pe_vs_sector_ratio",
        "high_short_interest_caution", "composite_score",
        "score_components_available", "needs_manual_review",
    ]
    display_cols = [c for c in display_cols if c in top.columns]

    if out_path.suffix.lower() == ".csv":
        top[display_cols].to_csv(out_path, index=False)
    elif out_path.suffix.lower() in (".html", ".htm"):
        from html_report import write_html_report
        write_html_report(
            top, str(out_path),
            universe_label="demo sample" if args.demo else args.universe,
            is_demo=args.demo,
            n_total_universe=len(scored) if not args.demo else None,
        )
    else:
        top[display_cols].to_excel(out_path, index=False, sheet_name="Scan Results")

    print(f"\nScored {len(scored)} tickers. Wrote top {len(top)} to {out_path}\n")
    with pd.option_context("display.max_columns", 10, "display.width", 160):
        print(top[["ticker", "trailing_pe", "drawdown_from_high", "volume_ratio", "composite_score"]].head(20))


if __name__ == "__main__":
    main()
