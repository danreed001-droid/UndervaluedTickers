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
    python scanner.py --universe sp500 --workers 16 --top 40 -o docs/index.html
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
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

log = logging.getLogger("scanner")

WEIGHTS = {
    "pe_cheap": 2.0,
    "historical_cheap": 2.0,
    "low_volume": 2.0,
    "pullback_10_20": 2.0,
    "peg": 1.0,
    "pe_trend": 1.0,
    "fcf_yield": 1.0,
    "debt_equity": 1.0,
    "roe": 1.0,
    "pe_vs_sector": 1.0,
    "payout_sustainable": 1.0,
}

RECENT_VOL_DAYS = 15
NORMAL_VOL_DAYS = 252
VOL_RATIO_FLAG = 0.85
PULLBACK_LO, PULLBACK_HI = 0.10, 0.20

NASDAQ_LISTED_URL = "https://ftp.nasdaqtrader.com/dynamiclookup/nasdaqlisted.txt"
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp500_tickers() -> list[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    resp = requests.get(SP500_WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    tickers = df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
    return sorted(tickers.unique().tolist())


def get_nasdaq_composite_tickers() -> list[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    resp = requests.get(NASDAQ_LISTED_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    lines = [l for l in resp.text.splitlines() if l and not l.startswith("File Creation")]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|")
    df = df[(df["Test Issue"] == "N") & (df["ETF"] == "N")]
    tickers = df["Symbol"].astype(str).str.strip()
    tickers = tickers[~tickers.str.contains(r"[\.\$]", regex=True)]
    return sorted(tickers.unique().tolist())


def build_universe(name: str) -> list[str]:
    if name == "sp500":
        return get_sp500_tickers()
    if name == "nasdaq":
        return get_nasdaq_composite_tickers()
    if name == "both":
        return sorted(set(get_sp500_tickers()) | set(get_nasdaq_composite_tickers()))
    raise ValueError(f"unknown universe {name!r}")


@dataclass
class FetchResult:
    ticker: str
    ok: bool
    error: str | None = None
    data: dict = field(default_factory=dict)


def fetch_one(ticker: str, retries: int = 3, backoff: float = 1.5) -> FetchResult:
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
    drawdown = (fifty2wk_high - current_price) / fifty2wk_high if (current_price and fifty2wk_high and fifty2wk_high > 0) else None

    recent_vol = float(vol.tail(RECENT_VOL_DAYS).mean()) if len(vol) >= RECENT_VOL_DAYS else None
    baseline_slice = vol.iloc[:-RECENT_VOL_DAYS] if len(vol) > RECENT_VOL_DAYS else vol
    normal_vol = float(baseline_slice.tail(NORMAL_VOL_DAYS).mean()) if len(baseline_slice) else None
    vol_ratio = (recent_vol / normal_vol) if (recent_vol and normal_vol) else None

    eps = info.get("trailingEps")
    hist_pe_median = None
    if eps and eps > 0 and len(close) >= 60:
        hist_pe_series = (close / eps).replace([np.inf, -np.inf], np.nan).dropna()
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


def fetch_universe(tickers: list[str], workers: int, sleep_between: float, checkpoint_path: Path, resume: bool) -> pd.DataFrame:
    done_rows: dict[str, dict] = {}
    if resume and checkpoint_path.exists():
        existing = pd.read_csv(checkpoint_path)
        done_rows = {r["ticker"]: r for r in existing.to_dict("records")}

    todo = [t for t in tickers if t not in done_rows]
    rows = list(done_rows.values())

    def flush():
        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_one, t): t for t in todo}
        for i, fut in enumerate(tqdm(cf.as_completed(futures), total=len(futures), desc="scanning")):
            res = fut.result()
            if res.ok:
                rows.append(res.data)
            if sleep_between:
                time.sleep(sleep_between)
            if i % 50 == 0:
                flush()
    flush()
    return pd.DataFrame(rows)


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

    df["sector_median_pe"] = df.groupby("sector")["trailing_pe"].transform("median") if "sector" in df.columns else df["trailing_pe"].median()
    df["pe_vs_sector_ratio"] = df["trailing_pe"] / df["sector_median_pe"]
    df["pe_vs_history_ratio"] = df["trailing_pe"] / df["historical_median_pe"]

    scores = pd.DataFrame(index=df.index)
    scores["pe_cheap"] = _lower_is_better(df["trailing_pe"])
    scores["historical_cheap"] = _lower_is_better(df["pe_vs_history_ratio"])
    scores["low_volume"] = _lower_is_better(df["volume_ratio"])

    def pullback_score(dd):
        if pd.isna(dd): return np.nan
        if PULLBACK_LO <= dd <= PULLBACK_HI: return 1.0
        if (PULLBACK_LO - 0.05) <= dd < PULLBACK_LO or PULLBACK_HI < dd <= (PULLBACK_HI + 0.10): return 0.5
        return 0.0
    scores["pullback_10_20"] = df["drawdown_from_high"].apply(pullback_score)
    scores["peg"] = _lower_is_better(df["peg"], good_q=0.33, ok_q=0.66)
    scores["pe_trend"] = np.where(df["forward_pe"].notna() & df["trailing_pe"].notna(), (df["forward_pe"] < df["trailing_pe"]).astype(float), np.nan)
    scores["fcf_yield"] = _higher_is_better(df["fcf_yield"])
    scores["debt_equity"] = _lower_is_better(df["debt_equity"])
    scores["roe"] = _higher_is_better(df["roe"])
    scores["pe_vs_sector"] = _lower_is_better(df["pe_vs_sector_ratio"])
    scores["payout_sustainable"] = _lower_is_better(df["payout_ratio"].where(df["dividend_yield"] > 0), good_q=0.33, ok_q=0.66)

    weight_row = pd.Series(WEIGHTS)
    weighted = scores[weight_row.index] * weight_row
    weight_available = scores[weight_row.index].notna() * weight_row
    df["composite_score"] = (weighted.sum(axis=1) / weight_available.sum(axis=1).replace(0, np.nan)) * 100

    for col in scores.columns:
        df[f"flag_{col}"] = scores[col]

    # Generate narrative summary blurbs
    def make_blurb(r):
        reasons = []
        if pd.notna(r.get("flag_historical_cheap")) and r["flag_historical_cheap"] >= 1.0:
            reasons.append("Trading significantly below its 5-yr historical P/E median.")
        if pd.notna(r.get("drawdown_from_high")) and 0.10 <= r["drawdown_from_high"] <= 0.20:
            reasons.append(f"Healthy pullback ({r['drawdown_from_high']*100:.1f}% off 52w high).")
        if pd.notna(r.get("volume_ratio")) and r["volume_ratio"] < VOL_RATIO_FLAG:
            reasons.append("Recent volume is running quieter than normal baseline.")
        if pd.notna(r.get("fcf_yield")) and r["fcf_yield"] > 0.08:
            reasons.append(f"Attractive Free Cash Flow yield ({r['fcf_yield']*100:.1f}%).")
        if not reasons:
            reasons.append("Meets multiple cross-sectional valuation checks.")
        return " ".join(reasons)

    df["score_blurb"] = df.apply(make_blurb, axis=1)
    df["needs_manual_review"] = "Check Form 4 Insiders & market thesis catalysts"
    return df.sort_values("composite_score", ascending=False)


def write_html_report(df: pd.DataFrame, out_path: str, universe_label: str, n_total_universe: int | None):
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Undervalued Tickers Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; background: #0f172a; color: #f8fafc; }}
        .header {{ background: #1e293b; padding: 25px 30px; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }}
        h1 {{ margin: 0; font-size: 24px; color: #38bdf8; }}
        .subtitle {{ color: #94a3b8; font-size: 13px; margin-top: 5px; }}
        .stats-bar {{ display: flex; gap: 20px; padding: 20px 30px; background: #111827; }}
        .stat-card {{ background: #1f2937; padding: 15px 20px; border-radius: 8px; border: 1px solid #374151; flex: 1; }}
        .stat-val {{ font-size: 22px; font-weight: bold; color: #38bdf8; }}
        .stat-label {{ font-size: 12px; color: #9ca3af; text-transform: uppercase; margin-top: 4px; }}
        
        /* Column Header Index Legend */
        .legend-box {{ background: #1e293b; margin: 20px 30px; padding: 15px 20px; border-radius: 8px; border: 1px solid #334155; }}
        .legend-box h3 {{ margin: 0 0 10px 0; font-size: 14px; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.5px; }}
        .legend-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px 15px; font-size: 12px; color: #cbd5e1; }}
        .legend-item b {{ color: #38bdf8; }}

        .table-container {{ margin: 20px 30px; background: #1e293b; border-radius: 8px; overflow-x: auto; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }}
        table {{ border-collapse: collapse; width: 100%; white-space: nowrap; }}
        th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid #334155; font-size: 13px; }}
        th {{ background: #0f172a; color: #cbd5e1; position: sticky; top: 0; z-index: 2; font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }}
        tr:hover {{ background: #273548; }}
        .score-badge {{ background: #0369a1; color: #e0f2fe; padding: 4px 8px; border-radius: 6px; font-weight: bold; text-align: center; }}
        .blurb-text {{ font-size: 12px; color: #94a3b8; max-width: 350px; white-space: normal; line-height: 1.4; }}
        .flag-good {{ color: #4ade80; background: rgba(74, 222, 128, 0.1); text-align: center; font-weight: bold; border-radius: 4px; }}
        .flag-mid {{ color: #facc15; background: rgba(250, 204, 21, 0.1); text-align: center; border-radius: 4px; }}
        .flag-bad {{ color: #f87171; background: rgba(248, 113, 113, 0.1); text-align: center; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Undervalued Tickers Executive Dashboard</h1>
            <div class="subtitle">Universe: <b>{universe_label.upper()}</b> | Total Analyzed: {n_total_universe if n_total_universe else len(df)} Tickers</div>
        </div>
    </div>

    <div class="stats-bar">
        <div class="stat-card">
            <div class="stat-val">{len(df)}</div>
            <div class="stat-label">Shortlisted Tickers</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{df['composite_score'].max():.1f}</div>
            <div class="stat-label">Max Composite Score</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{df['composite_score'].median():.1f}</div>
            <div class="stat-label">Median Composite Score</div>
        </div>
    </div>

    <div class="legend-box">
        <h3>Column Header Index & Definitions</h3>
        <div class="legend-grid">
            <div class="legend-item"><b>Score:</b> Composite 0-100 valuation rating (weighted 2x core criteria, 1x secondary).</div>
            <div class="legend-item"><b>P/E:</b> Trailing Price-to-Earnings ratio.</div>
            <div class="legend-item"><b>Hist P/E Med:</b> Stock's own ~5 year historical median P/E baseline.</div>
            <div class="legend-item"><b>Vol Ratio:</b> Recent 15-day volume divided by 252-day normal average.</div>
            <div class="legend-item"><b>Drawdown:</b> Percentage distance below 52-week high price peak.</div>
            <div class="legend-item"><b>PEG:</b> Price/Earnings-to-Growth valuation ratio.</div>
            <div class="legend-item"><b>Fwd P/E:</b> Forward expected earnings P/E ratio.</div>
            <div class="legend-item"><b>FCF Yield:</b> Free cash flow divided by total market capitalization.</div>
            <div class="legend-item"><b>D/E:</b> Total debt-to-equity leverage ratio.</div>
            <div class="legend-item"><b>ROE:</b> Return on equity profitability metric.</div>
            <div class="legend-item"><b>Rationale Blurb:</b> Automated breakdown explaining why the stock scored well.</div>
        </div>
    </div>

    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Company</th>
                    <th>Sector</th>
                    <th>Price</th>
                    <th style="text-align: center;">Score</th>
                    <th>P/E</th>
                    <th>Hist P/E Med</th>
                    <th>Vol Ratio</th>
                    <th>Drawdown</th>
                    <th>PEG</th>
                    <th>Fwd P/E</th>
                    <th>FCF Yield</th>
                    <th>D/E</th>
                    <th>ROE</th>
                    <th>Why It Scored This Way (Rationale)</th>
                    <th>Manual Check</th>
                </tr>
            </thead>
            <tbody>
"""

    def render_flag(val):
        if pd.isna(val): return "<td>-</td>"
        vf = float(val)
        cls = "flag-good" if vf >= 1.0 else ("flag-mid" if vf >= 0.5 else "flag-bad")
        return f'<td class="{cls}">{vf:.1f}</td>'

    for _, r in df.iterrows():
        t = r.get("ticker", "")
        comp = r.get("company", "") or ""
        sec = r.get("sector", "") or ""
        price = f"${r['price']:.2f}" if pd.notna(r.get("price")) else "N/A"
        score = f"{r.get('composite_score', 0):.1f}"
        pe = f"{r.get('trailing_pe', np.nan):.1f}" if pd.notna(r.get("trailing_pe")) else "N/A"
        h_med = f"{r.get('historical_median_pe', np.nan):.1f}" if pd.notna(r.get("historical_median_pe")) else "N/A"
        v_rat = f"{r.get('volume_ratio', np.nan):.2f}" if pd.notna(r.get("volume_ratio")) else "N/A"
        dd = f"{r.get('drawdown_from_high', 0)*100:.1f}%" if pd.notna(r.get("drawdown_from_high")) else "N/A"
        peg = f"{r.get('peg', np.nan):.2f}" if pd.notna(r.get("peg")) else "N/A"
        fwd = f"{r.get('forward_pe', np.nan):.1f}" if pd.notna(r.get("forward_pe")) else "N/A"
        fcf = f"{r.get('fcf_yield', 0)*100:.1f}%" if pd.notna(r.get("fcf_yield")) else "N/A"
        de = f"{r.get('debt_equity', np.nan):.2f}" if pd.notna(r.get("debt_equity")) else "N/A"
        roe = f"{r.get('roe', 0)*100:.1f}%" if pd.notna(r.get("roe")) else "N/A"
        blurb = r.get("score_blurb", "")
        review = r.get("needs_manual_review", "")

        html_content += f"""
            <tr>
                <td><b>{t}</b></td>
                <td>{comp}</td>
                <td>{sec}</td>
                <td>{price}</td>
                <td><div class="score-badge">{score}</div></td>
                <td>{pe}</td>
                <td>{h_med}</td>
                <td>{v_rat}</td>
                <td>{dd}</td>
                <td>{peg}</td>
                <td>{fwd}</td>
                <td>{fcf}</td>
                <td>{de}</td>
                <td>{roe}</td>
                <td><div class="blurb-text">{blurb}</div></td>
                <td style="color: #94a3b8; font-size: 11px;">{review}</td>
            </tr>
"""

    html_content += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    Path(out_path).write_text(html_content, encoding="utf-8")


def demo_dataframe() -> pd.DataFrame:
    rows = [
        dict(ticker="CMCSA", company="Comcast Corporation", sector="Communication Services", market_cap=92.90e9, price=26.18, trailing_pe=8.47, forward_pe=7.45, peg=None, pb=1.03, debt_equity=1.00, roe=0.1149, dividend_yield=0.0504, payout_ratio=0.4273, short_pct_float=0.021, fcf_yield=0.09, fifty2wk_high=33.5, drawdown_from_high=0.219, recent_avg_volume=30500969, normal_avg_volume=35252514, volume_ratio=0.865, historical_median_pe=11.2),
        dict(ticker="CNC", company="Centene Corporation", sector="Healthcare", market_cap=32.855e9, price=66.54, trailing_pe=12.4, forward_pe=14.44, peg=0.38, pb=1.48, debt_equity=0.71, roe=0.10, dividend_yield=None, payout_ratio=None, short_pct_float=0.0296, fcf_yield=0.05, fifty2wk_high=78.0, drawdown_from_high=0.147, recent_avg_volume=5017450, normal_avg_volume=5803495, volume_ratio=0.865, historical_median_pe=20.5),
        dict(ticker="APA", company="APA Corporation", sector="Energy", market_cap=12.778e9, price=36.15, trailing_pe=8.57, forward_pe=9.58, peg=None, pb=2.02, debt_equity=0.49, roe=0.2666, dividend_yield=0.0247, payout_ratio=0.2117, short_pct_float=0.0864, fcf_yield=0.11, fifty2wk_high=44.0, drawdown_from_high=0.178, recent_avg_volume=5506362, normal_avg_volume=5860269, volume_ratio=0.940, historical_median_pe=9.8),
    ]
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--universe", choices=["sp500", "nasdaq", "both"], default="both")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--sleep", type=float, default=0.05)
    p.add_argument("--checkpoint", default="scan_checkpoint.csv")
    p.add_argument("--resume", action="store_true")
    p.add_argument("-o", "--output", default="docs/index.html")
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--demo", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    if args.demo:
        print("Running in --demo mode...")
        raw = demo_dataframe()
    else:
        tickers = build_universe(args.universe)
        if args.limit:
            random.shuffle(tickers)
            tickers = tickers[: args.limit]
        raw = fetch_universe(tickers, workers=args.workers, sleep_between=args.sleep, checkpoint_path=Path(args.checkpoint), resume=args.resume)

    if raw.empty:
        print("No data fetched.", file=sys.stderr)
        sys.exit(1)

    scored = score_universe(raw)
    top = scored.head(args.top)
    write_html_report(top, args.output, universe_label="demo sample" if args.demo else args.universe, n_total_universe=len(scored) if not args.demo else None)
    print(f"\nDashboard generated successfully at: {args.output}\n")


if __name__ == "__main__":
    main()
