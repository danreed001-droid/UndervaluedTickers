#!/usr/bin/env python3
"""
scanner.py
===========

A P/E + historical-valuation + volume + pullback screener for the S&P 500
and Nasdaq Composite, with a transparent, self-relative composite score.

USAGE
-----
    # Quick smoke test with fabricated data, no network needed:
    python scanner.py --demo

    # Real run, S&P 500 only, 16 worker threads, save to docs/index.html:
    python scanner.py --universe sp500 --workers 16 --top 150 -o docs/index.html

    # Full universe (S&P 500 + Nasdaq Composite), resume a prior run:
    python scanner.py --universe both --resume -o full_scan.html
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
    """Scrape the current S&P 500 constituent list from Wikipedia with a valid User-Agent."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    resp = requests.get(SP500_WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    
    tables = pd.read_html(resp.text)
    df = tables[0]
    tickers = df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
    return sorted(tickers.unique().tolist())


def get_nasdaq_composite_tickers() -> list[str]:
    """
    Approximate the Nasdaq Composite with nasdaqtrader.com's listed-securities
    file, filtered to common/ordinary shares (drops ETFs and test issues).
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    resp = requests.get(NASDAQ_LISTED_URL, headers=headers, timeout=30)
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
    import yfinance as yf

    last_err = None
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            hist = t.history(period="5y", interval="1d", auto_adjust=False)
            return FetchResult(ticker, True, data=_extract_metrics(ticker, info, hist))
        except Exception as exc:
            last_err = str(exc)
            time.sleep(backoff ** attempt + random.random())
    return FetchResult(ticker, False, error=last_err)


def _extract_metrics(ticker: str, info: dict, hist: pd.DataFrame) -> dict:
    close = hist["Close"].dropna() if not hist.empty else pd.Series(dtype=float)
    vol = hist["Volume"].dropna() if not hist.empty else pd.Series(dtype=float)

    current_price = float(close.iloc[-1]) if len(close) else info.get("currentPrice")
    fifty2wk_high = float(close.tail(252).max()) if len(close) >= 5 else info.get("fiftyTwoWeekHigh")
    drawdown = None
    if current_price and fifty2wk_high and fifty2wk_high > 0:
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
        debt_to_equity = debt_to_equity / 100.0

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
# Scoring
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

    if "sector" in df.columns:
        df["sector_median_pe"] = df.groupby("sector")["trailing_pe"].transform("median")
    else:
        df["sector_median_pe"] = df["trailing_pe"].median()
    df["pe_vs_sector_ratio"] = df["trailing_pe"] / df["sector_median_pe"]
    df["pe_vs_history_ratio"] = df["trailing_pe"] / df["historical_median_pe"]

    scores = pd.DataFrame(index=df.index)
    scores["pe_cheap"] = _lower_is_better(df["trailing_pe"])
    scores["historical_cheap"] = _lower_is_better(df["pe_vs_history_ratio"])
    scores["low_volume"] = _lower_is_better(df["volume_ratio"])

    def pullback_score(dd):
        if pd.isna(dd):
            return np.nan
        if PULLBACK_LO <= dd <= PULLBACK_HI:
            return 1.0
        if (PULLBACK_LO - 0.05) <= dd < PULLBACK_LO or PULLBACK_HI < dd <= (PULLBACK_HI + 0.10):
            return 0.5
        return 0.0
    scores["pullback_10_20"] = df["drawdown_from_high"].apply(pullback_score)

    scores["peg"] = _lower_is_better(df["peg"], good_q=0.33, ok_q=0.66)
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
# HTML Report Exporter (Built-in)
# ---------------------------------------------------------------------------
def write_html_report(df: pd.DataFrame, out_path: str, universe_label: str, is_demo: bool, n_total_universe: int | None):
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Undervalued Tickers Scan Results</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px; background: #f8f9fa; color: #333; }}
        h1 {{ margin-bottom: 5px; }}
        .subtitle {{ color: #666; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #dee2e6; font-size: 14px; }}
        th {{ background: #343a40; color: #fff; position: sticky; top: 0; }}
        tr:hover {{ background: #f1f3f5; }}
        .badge {{ padding: 3px 6px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
        .badge-yes {{ background: #d4edda; color: #155724; }}
        .badge-no {{ background: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <h1>Undervalued Tickers Screener</h1>
    <div class="subtitle">Universe: <b>{universe_label}</b> | Total Scanned: {n_total_universe if n_total_universe else len(df)}</div>
    <table>
        <thead>
            <tr>
                <th>Ticker</th>
                <th>Company</th>
                <th>Sector</th>
                <th>Price</th>
                <th>P/E</th>
                <th>Fwd P/E</th>
                <th>Drawdown</th>
                <th>Vol Ratio</th>
                <th>Score</th>
                <th>Manual Review</th>
            </tr>
        </thead>
        <tbody>
"""
    for _, row in df.iterrows():
        ticker = row.get("ticker", "")
        company = row.get("company", "") or ""
        sector = row.get("sector", "") or ""
        price = f"${row['price']:.2f}" if pd.notna(row.get("price")) else "N/A"
        pe = f"{row['trailing_pe']:.1f}" if pd.notna(row.get("trailing_pe")) else "N/A"
        fwd_pe = f"{row['forward_pe']:.1f}" if pd.notna(row.get("forward_pe")) else "N/A"
        dd = f"{row['drawdown_from_high']*100:.1f}%" if pd.notna(row.get("drawdown_from_high")) else "N/A"
        vol_r = f"{row['volume_ratio']:.2f}" if pd.notna(row.get("volume_ratio")) else "N/A"
        score = f"{row['composite_score']:.1f}" if pd.notna(row.get("composite_score")) else "0.0"
        review = row.get("needs_manual_review", "")

        html_content += f"""
            <tr>
                <td><b>{ticker}</b></td>
                <td>{company}</td>
                <td>{sector}</td>
                <td>{price}</td>
                <td>{pe}</td>
                <td>{fwd_pe}</td>
                <td>{dd}</td>
                <td>{vol_r}</td>
                <td><b>{score}</b></td>
                <td>{review}</td>
            </tr>
"""

    html_content += """
        </tbody>
    </table>
</body>
</html>
"""
    Path(out_path).write_text(html_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------
def demo_dataframe() -> pd.DataFrame:
    rows = [
        dict(ticker="CMCSA", company="Comcast Corporation", sector="Communication Services", market_cap=92.90e9, price=26.18, trailing_pe=8.47, forward_pe=7.45, peg=None, pb=1.03, debt_equity=1.00, roe=0.1149, dividend_yield=0.0504, payout_ratio=0.4273, short_pct_float=0.021, fcf_yield=0.09, fifty2wk_high=33.5, drawdown_from_high=0.219, recent_avg_volume=30500969, normal_avg_volume=35252514, volume_ratio=0.865, historical_median_pe=11.2),
        dict(ticker="CNC", company="Centene Corporation", sector="Healthcare", market_cap=32.855e9, price=66.54, trailing_pe=None, forward_pe=14.44, peg=0.38, pb=1.48, debt_equity=0.71, roe=-0.2036, dividend_yield=None, payout_ratio=None, short_pct_float=0.0296, fcf_yield=0.05, fifty2wk_high=78.0, drawdown_from_high=0.147, recent_avg_volume=5017450, normal_avg_volume=5803495, volume_ratio=0.865, historical_median_pe=20.5),
        dict(ticker="APA", company="APA Corporation", sector="Energy", market_cap=12.778e9, price=36.15, trailing_pe=8.57, forward_pe=9.58, peg=None, pb=2.02, debt_equity=0.49, roe=0.2666, dividend_yield=0.0247, payout_ratio=0.2117, short_pct_float=0.0864, fcf_yield=0.11, fifty2wk_high=44.0, drawdown_from_high=0.178, recent_avg_volume=5506362, normal_avg_volume=5860269, volume_ratio=0.940, historical_median_pe=9.8),
        dict(ticker="CHTR", company="Charter Communications", sector="Communication Services", market_cap=20.63e9, price=154.27, trailing_pe=4.01, forward_pe=3.58, peg=0.27, pb=1.09, debt_equity=4.42, roe=0.2720, dividend_yield=None, payout_ratio=None, short_pct_float=0.2925, fcf_yield=0.14, fifty2wk_high=270.0, drawdown_from_high=0.429, recent_avg_volume=2998008, normal_avg_volume=3458122, volume_ratio=0.867, historical_median_pe=6.0),
        dict(ticker="DVN", company="Devon Energy Corporation", sector="Energy", market_cap=49.291e9, price=42.74, trailing_pe=9.83, forward_pe=8.85, peg=0.98, pb=1.26, debt_equity=0.28, roe=0.1152, dividend_yield=0.0279, payout_ratio=0.2744, short_pct_float=0.0265, fcf_yield=0.10, fifty2wk_high=51.0, drawdown_from_high=0.162, recent_avg_volume=9694035, normal_avg_volume=14099226, volume_ratio=0.688, historical_median_pe=10.9),
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
        print("Running in --demo mode: scoring sample tickers, no network calls made.\n")
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
