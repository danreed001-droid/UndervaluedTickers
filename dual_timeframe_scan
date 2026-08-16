#!/usr/bin/env python3
"""
dual_timeframe_scan.py
=======================
A scannable watchlist report: every ticker in watchlists.GROUPS (S&P 500,
a fixed ETF/FX/crypto watchlist, and 15 themed VSA groups -- 533 unique
tickers total), one row each, with two compact candlestick charts side by
side -- last 50 hourly bars and the last ~365 daily bars -- plus a
click-to-filter group chip bar so you can narrow down to just one theme.

Design notes (see README for the fuller version):
  - Bulk chunked yf.download() calls (not one yf.Ticker() per ticker) --
    matches scan_volume.py's convention in this repo, and is dramatically
    fewer HTTP round-trips at this scale (533 tickers).
  - A bad/delisted/renamed ticker never aborts the run: every chunk-level
    and per-ticker extraction is wrapped so one failure just gets logged
    and skipped, per "don't fail the whole process."
  - Charts are inline SVG, not Plotly -- at 533 rows x 2 charts, a full
    interactive Plotly figure per chart would bloat the page far more than
    this scannable-list format calls for (plot_vsa2d.py's one-big-chart-
    per-ticker style is the right tool for a deep dive on a handful of
    names, not a 533-row watchlist).

USAGE
-----
    # Offline smoke test, no network needed (synthetic candles):
    python scripts/dual_timeframe_scan.py --demo -o /tmp/dual_demo.html

    # Real run:
    python scripts/dual_timeframe_scan.py -o reports/dual_timeframe_latest.html
"""
from __future__ import annotations

import argparse
import html
import math
import random
import sys
import time
from datetime import date, datetime

import watchlists as wl

CHUNK_SIZE_DEFAULT = 150
HOURLY_PERIOD = "1y"    # yfinance's 1h-interval history is capped ~730d; 1y is safely within it
HOURLY_INTERVAL = "1h"
HOURLY_BARS_SHOWN = 50  # "last 50 candles"
DAILY_PERIOD = "1y"     # ~365 calendar days -- the literal ask, not watchlists.TF_MAP's "max"
DAILY_INTERVAL = "1d"

GOOD = "#3fb950"      # matches plot_vsa2d.py's green-ish buyer color family, GitHub-dark palette
CRITICAL = "#f85149"  # matches plot_vsa2d.py's red seller color family
MUTED = "#8b949e"


# ─────────────────────────────────────────────────────────────────────────
# Fetching -- real (chunked yf.download) and synthetic (--demo) paths
# ─────────────────────────────────────────────────────────────────────────

def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _extract_bars(df, ticker: str, chunk_len: int) -> list[dict]:
    """Pull one ticker's OHLC rows out of a (possibly multi-ticker) yf.download
    result. Returns [] rather than raising on any shape/data surprise --
    callers treat an empty list as 'skip this ticker', never as a fatal error."""
    try:
        sub = df[ticker] if chunk_len > 1 else df
        sub = sub.dropna(subset=["Close"])
    except (KeyError, TypeError):
        return []
    if sub.empty:
        return []
    rows = []
    for ts, row in sub.iterrows():
        try:
            o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
        except (TypeError, ValueError):
            continue
        if any(math.isnan(v) for v in (o, h, l, c)):
            continue
        rows.append({"t": ts.isoformat(), "o": o, "h": h, "l": l, "c": c})
    return rows


def download_timeframe(tickers: list[str], period: str, interval: str, chunk_size: int,
                        label: str) -> tuple[dict[str, list[dict]], list[str]]:
    """Chunked bulk download for one timeframe across every ticker. Returns
    (bars_by_ticker, failed_tickers) -- failed tickers are logged and
    carried forward, never raised."""
    import yfinance as yf  # imported lazily so --demo needs no network lib import errors

    bars: dict[str, list[dict]] = {}
    failed: list[str] = []
    chunks = _chunks(tickers, chunk_size)
    for idx, chunk in enumerate(chunks, start=1):
        print(f"  [{label}] chunk {idx}/{len(chunks)} ({len(chunk)} tickers)", file=sys.stderr)
        data = None
        for attempt in range(3):
            try:
                data = yf.download(
                    tickers=chunk, period=period, interval=interval,
                    group_by="ticker", threads=True, progress=False, auto_adjust=True,
                )
                break
            except Exception as exc:  # noqa: BLE001 -- a chunk-level network hiccup, not fatal
                print(f"    [{label}] retry {attempt + 1} after error: {exc}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        if data is None:
            failed.extend(chunk)
            continue
        for ticker in chunk:
            rows = _extract_bars(data, ticker, len(chunk))
            if rows:
                bars[ticker] = rows
            else:
                failed.append(ticker)
        time.sleep(1)  # be polite between chunks, matches scan_volume.py
    return bars, failed


def _demo_bars(ticker: str, seed: int, n: int, start_price: float) -> list[dict]:
    """Synthetic OHLC bars for --demo (no network). Seeded per-ticker so
    re-running --demo is reproducible."""
    rng = random.Random(seed)
    price = start_price
    rows = []
    for i in range(n):
        o = price
        drift = rng.uniform(-0.018, 0.018)
        c = max(o * (1 + drift), 0.01)
        h = max(o, c) * (1 + rng.uniform(0.0, 0.01))
        l = min(o, c) * (1 - rng.uniform(0.0, 0.01))
        rows.append({"t": f"bar{i}", "o": round(o, 4), "h": round(h, 4), "l": round(l, 4), "c": round(c, 4)})
        price = c
    return rows


def build_demo_dataset(tickers: list[str]) -> tuple[dict[str, list[dict]], dict[str, list[dict]], list[str]]:
    hourly, daily = {}, {}
    for i, t in enumerate(tickers):
        base = 20 + (i * 37) % 400  # spread starting prices out a bit
        hourly[t] = _demo_bars(t, seed=i, n=HOURLY_BARS_SHOWN + 10, start_price=base)
        daily[t] = _demo_bars(t, seed=i + 10_000, n=252, start_price=base)
    return hourly, daily, []


# ─────────────────────────────────────────────────────────────────────────
# Inline-SVG candlestick strip
# ─────────────────────────────────────────────────────────────────────────

def _candles_svg(bars: list[dict], width: int = 118, height: int = 36) -> str:
    if not bars or len(bars) < 2:
        return '<span class="muted small">&mdash;</span>'

    highs = [b["h"] for b in bars]
    lows = [b["l"] for b in bars]
    lo, hi = min(lows), max(highs)
    rng = (hi - lo) or 1.0
    n = len(bars)
    pad = 2.0
    slot = width / n
    body_w = max(1.2, min(slot * 0.62, 6.0))

    def y(v: float) -> float:
        return pad + (hi - v) / rng * (height - 2 * pad)

    parts = []
    for i, b in enumerate(bars):
        cx = slot * i + slot / 2
        up = b["c"] >= b["o"]
        color = "var(--good)" if up else "var(--critical)"
        y_h, y_l = y(b["h"]), y(b["l"])
        y_o, y_c = y(b["o"]), y(b["c"])
        top, bot = (y_o, y_c) if y_o < y_c else (y_c, y_o)
        if bot - top < 1.0:
            bot = top + 1.0
        parts.append(f'<line x1="{cx:.1f}" y1="{y_h:.1f}" x2="{cx:.1f}" y2="{y_l:.1f}" stroke="{color}" stroke-width="1"/>')
        parts.append(f'<rect x="{cx - body_w/2:.1f}" y="{top:.1f}" width="{body_w:.1f}" height="{bot-top:.1f}" fill="{color}"/>')

    first_open, last_close = bars[0]["o"], bars[-1]["c"]
    pct = (last_close / first_open - 1.0) * 100 if first_open else 0.0
    title = f"{n} bars: {first_open:.2f} → {last_close:.2f} ({pct:+.1f}%)"
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'class="candles" role="img" aria-label="{html.escape(title)}">'
        f'<title>{html.escape(title)}</title>{"".join(parts)}</svg>'
    )


# ─────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────

ROW_TEMPLATE = """
<tr data-groups="{groups_attr}">
  <td class="mono col-rank">{rank}</td>
  <td class="mono">{ticker}</td>
  <td class="groups-cell">{groups_html}</td>
  <td>{hourly_svg}</td>
  <td>{daily_svg}</td>
  <td class="num {delta_cls}">{delta_text}</td>
  <td class="statuscell"><span class="dot {dot_cls}"></span></td>
</tr>
"""

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dual-Timeframe Watchlist &mdash; {report_date}</title>
<style>
  :root {{
    --plane: #0d1117; --surface: #161b22; --ink: #f0f6fc; --ink2: #c9d1d9;
    --muted: #8b949e; --grid: #21262d; --border: #30363d;
    --series: #58a6ff; --good: {good}; --critical: {critical};
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--plane); color: var(--ink2); font-family: 'Courier New', monospace; }}
  .wrap {{ max-width: 1400px; margin: 0 auto; padding: 28px 20px 60px; }}
  h1 {{ font-size: 19px; color: var(--ink); margin: 0 0 4px; letter-spacing: 1px; }}
  .subtitle {{ font-size: 12px; color: var(--muted); margin: 0 0 20px; }}
  .banner {{
    background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--series);
    border-radius: 8px; padding: 12px 16px; font-size: 12.5px; color: var(--ink2); margin-bottom: 20px; line-height: 1.6;
  }}
  .filterbar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-bottom: 12px; }}
  .chip {{
    font: inherit; font-size: 11.5px; font-weight: 600; padding: 5px 12px; border-radius: 999px;
    border: 1px solid var(--border); background: var(--surface); color: var(--ink2); cursor: pointer; white-space: nowrap;
  }}
  .chip:hover {{ background: #1c2129; }}
  .chip.active {{ background: var(--series); border-color: var(--series); color: #0d1117; }}
  .filter-count {{ font-size: 11.5px; color: var(--muted); margin-left: auto; white-space: nowrap; }}

  .table-scroll {{ max-height: 78vh; overflow: auto; border-radius: 8px; border: 1px solid var(--border); }}
  table {{ width: 100%; border-collapse: collapse; background: var(--surface); }}
  thead th {{
    text-align: left; font-size: 10.5px; text-transform: uppercase; letter-spacing: .04em;
    color: var(--muted); padding: 8px 10px; border-bottom: 1px solid var(--border);
    cursor: pointer; user-select: none; white-space: nowrap; position: sticky; top: 0; background: var(--surface); z-index: 2;
  }}
  thead th:hover {{ color: var(--ink); }}
  thead th.sorted::after {{ content: " \\25BE"; }}
  thead th.nosort {{ cursor: default; }}
  thead th.nosort:hover {{ color: var(--muted); }}
  tbody td {{ padding: 6px 10px; border-bottom: 1px solid var(--grid); font-size: 12px; vertical-align: middle; }}
  tbody tr:hover {{ background: #1c2129; }}
  tbody tr:hover td.col-rank {{ background: #1c2129; }}
  td.num {{ font-variant-numeric: tabular-nums; text-align: right; font-weight: 700; }}
  td.mono {{ font-variant-numeric: tabular-nums; }}
  .muted {{ color: var(--muted); }}
  .small {{ font-size: 11px; }}
  .good {{ color: var(--good); }}
  .critical {{ color: var(--critical); }}
  .groups-cell {{ font-size: 10.5px; color: var(--muted); max-width: 220px; }}
  .col-rank {{ position: sticky; left: 0; z-index: 1; background: var(--surface); text-align: center; color: var(--muted); }}
  thead th.col-rank {{ z-index: 3; }}
  .statuscell {{ text-align: center; }}
  .dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; }}
  .dot.good {{ background: var(--good); }}
  .dot.critical {{ background: var(--critical); }}
  footer {{ margin-top: 20px; font-size: 11.5px; color: var(--muted); line-height: 1.6; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>&#9679; DUAL-TIMEFRAME WATCHLIST</h1>
  <p class="subtitle">Generated {generated_at} &middot; {n_shown} tickers &middot; hourly = last {hourly_bars} bars, daily = last ~365 days</p>

  <div class="banner">{banner}</div>

  <div class="filterbar" id="group-filter">
    {group_chips}
    <span class="filter-count" id="filter-count">Showing {n_shown} of {n_shown}</span>
  </div>

  <div class="table-scroll">
  <table id="results">
    <thead>
      <tr>
        <th data-type="skip" class="nosort col-rank">#</th>
        <th data-type="text">Ticker</th>
        <th data-type="text">Groups</th>
        <th data-type="skip" class="nosort">1h &middot; last {hourly_bars} candles</th>
        <th data-type="skip" class="nosort">1d &middot; last 365 days</th>
        <th data-type="num" class="sorted">Daily &Delta;%</th>
        <th data-type="skip" class="nosort">Up/Down</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  </div>

  <footer>{footer_note}</footer>
</div>
<script>
(function() {{
  const table = document.getElementById('results');
  const tbody = table.tBodies[0];
  const headers = table.tHead.rows[0].cells;
  let sortState = {{ col: 5, dir: -1 }};

  function renumberVisibleRanks() {{
    let n = 0;
    Array.from(tbody.rows).forEach(r => {{
      if (r.style.display === 'none') return;
      n++;
      const rankCell = r.cells[0];
      if (rankCell) rankCell.textContent = n;
    }});
  }}

  function cellValue(row, idx, type) {{
    const cell = row.cells[idx];
    if (type === 'num') {{
      const n = parseFloat(String(cell.textContent).replace(/[^0-9.\\-]/g, ''));
      return isNaN(n) ? -Infinity : n;
    }}
    return cell.textContent.trim().toLowerCase();
  }}

  function sortBy(idx) {{
    const type = headers[idx].getAttribute('data-type');
    if (type === 'skip') return;
    const dir = (sortState.col === idx) ? -sortState.dir : -1;
    sortState = {{ col: idx, dir }};
    const rows = Array.from(tbody.rows);
    rows.sort((a, b) => {{
      const av = cellValue(a, idx, type), bv = cellValue(b, idx, type);
      if (av < bv) return -1 * dir;
      if (av > bv) return 1 * dir;
      return 0;
    }});
    rows.forEach(r => tbody.appendChild(r));
    renumberVisibleRanks();
    Array.from(headers).forEach(h => h.classList.remove('sorted'));
    headers[idx].classList.add('sorted');
  }}

  Array.from(headers).forEach((h, idx) => h.addEventListener('click', () => sortBy(idx)));

  const filterbar = document.getElementById('group-filter');
  if (filterbar) {{
    const chips = Array.from(filterbar.querySelectorAll('.chip'));
    const allChip = filterbar.querySelector('.chip-all');
    const countEl = document.getElementById('filter-count');
    const rows = Array.from(tbody.rows);
    const total = rows.length;

    function applyFilter(group) {{
      rows.forEach(r => {{
        const groups = (r.getAttribute('data-groups') || '').split('|');
        const match = !group || groups.includes(group);
        r.style.display = match ? '' : 'none';
      }});
      renumberVisibleRanks();
      if (countEl) {{
        const shown = rows.filter(r => r.style.display !== 'none').length;
        countEl.textContent = group ? `Showing ${{shown}} of ${{total}} — ${{group}}` : `Showing ${{total}} of ${{total}}`;
      }}
    }}

    chips.forEach(chip => {{
      chip.addEventListener('click', () => {{
        const group = chip.getAttribute('data-group');
        const alreadyActive = chip.classList.contains('active');
        chips.forEach(c => c.classList.remove('active'));
        if (!group || alreadyActive) {{
          if (allChip) allChip.classList.add('active');
          applyFilter('');
        }} else {{
          chip.classList.add('active');
          applyFilter(group);
        }}
      }});
    }});
  }}
}})();
</script>
</body>
</html>
"""


def _group_chips_html(groups: list[str]) -> str:
    chips = ['<button type="button" class="chip chip-all active" data-group="">All groups</button>']
    for g in groups:
        g_esc = html.escape(g)
        chips.append(f'<button type="button" class="chip" data-group="{g_esc}">{g_esc}</button>')
    return "\n    ".join(chips)


def build_report_html(hourly: dict[str, list[dict]], daily: dict[str, list[dict]],
                       ticker_groups: dict[str, list[str]], failed: list[str], is_demo: bool) -> str:
    rows_html = []
    shown_tickers = sorted(t for t in ticker_groups if t in hourly or t in daily)

    for i, ticker in enumerate(shown_tickers, start=1):
        h_bars = hourly.get(ticker, [])[-HOURLY_BARS_SHOWN:]
        d_bars = daily.get(ticker, [])
        groups = ticker_groups.get(ticker, [])

        if len(d_bars) >= 2:
            delta = (d_bars[-1]["c"] / d_bars[-2]["c"] - 1.0) * 100
        elif len(d_bars) == 1:
            delta = 0.0
        else:
            delta = None

        up = (delta is not None) and delta >= 0
        rows_html.append(
            ROW_TEMPLATE.format(
                rank=i,
                groups_attr=html.escape("|".join(groups)),
                ticker=html.escape(ticker),
                groups_html=html.escape(", ".join(groups)),
                hourly_svg=_candles_svg(h_bars),
                daily_svg=_candles_svg(d_bars),
                delta_cls=("good" if up else "critical") if delta is not None else "muted",
                delta_text=f"{delta:+.1f}%" if delta is not None else "&mdash;",
                dot_cls="good" if up else "critical",
            )
        )

    n_shown = len(shown_tickers)
    n_total_requested = len(ticker_groups)
    n_failed = len(set(failed))

    banner = (
        "DEMO DATA &mdash; synthetic candles for a small sample of tickers, generated offline (no network "
        "calls). Run <code>python scripts/dual_timeframe_scan.py -o reports/dual_timeframe_latest.html</code> "
        "with real internet access for a live scan of all 533 tickers."
        if is_demo else
        "Live scan. Charts are unscaled per-row (each chart's own high/low), so they show shape/direction, "
        "not comparable absolute price across rows &mdash; hover a chart for exact open/close/% for that window."
    )

    footer_note = (
        f"{n_shown} of {n_total_requested} requested tickers rendered"
        + (f"; {n_failed} skipped (no data returned by Yahoo Finance for either timeframe &mdash; "
           f"delisted, renamed, or a transient fetch failure, not a bug in this report)." if n_failed else ".")
        + " Candles: green body = close &ge; open for that bar, red = close &lt; open. "
          "Daily &Delta;% is the latest daily bar's close vs. the prior daily bar's close."
    )

    return PAGE_TEMPLATE.format(
        good=GOOD, critical=CRITICAL,
        report_date=date.today().isoformat(),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_shown=n_shown,
        hourly_bars=HOURLY_BARS_SHOWN,
        banner=banner,
        group_chips=_group_chips_html(sorted(wl.GROUPS.keys())),
        rows="\n".join(rows_html),
        footer_note=footer_note,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true", help="Synthetic candles, no network calls.")
    p.add_argument("--limit", type=int, default=None, help="Only scan this many tickers (random sample) -- for testing.")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT)
    p.add_argument("-o", "--output", default="reports/dual_timeframe_latest.html")
    args = p.parse_args()

    ticker_groups = wl.ticker_groups_map()
    tickers = sorted(ticker_groups.keys())
    if args.limit:
        rng = random.Random(0)
        tickers = sorted(rng.sample(tickers, min(args.limit, len(tickers))))
        ticker_groups = {t: g for t, g in ticker_groups.items() if t in tickers}

    if args.demo:
        print(f"Running in --demo mode with {len(tickers)} tickers (no network calls)...")
        demo_tickers = tickers[: args.limit or 24]
        ticker_groups = {t: g for t, g in ticker_groups.items() if t in demo_tickers}
        hourly, daily, failed = build_demo_dataset(demo_tickers)
    else:
        print(f"Fetching {len(tickers)} tickers x 2 timeframes (hourly, daily)...", file=sys.stderr)
        hourly, failed_h = download_timeframe(tickers, HOURLY_PERIOD, HOURLY_INTERVAL, args.chunk_size, "1h")
        daily, failed_d = download_timeframe(tickers, DAILY_PERIOD, DAILY_INTERVAL, args.chunk_size, "1d")
        # Only truly missing on BOTH timeframes counts as "failed" for the footer note --
        # a ticker with just one timeframe still gets a row (the other chart shows "-").
        failed = [t for t in set(failed_h) & set(failed_d)]

    html_str = build_report_html(hourly, daily, ticker_groups, failed, is_demo=args.demo)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"\nWrote {args.output} ({len(ticker_groups)} tickers requested, {len(set(failed))} fully failed)\n")


if __name__ == "__main__":
    main()
