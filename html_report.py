"""
html_report.py
===============
Renders scanner results as a single self-contained, dependency-free HTML
file (no external CSS/JS/fonts) with a click-to-sort table, status badges
for the low-volume / pullback flags, and a mini bar for the composite
score. Colors follow Anthropic's dataviz status/ink palette so it reads
consistently in light and dark mode.

Kept separate from scanner.py so the scoring logic and the presentation
layer don't get tangled.
"""

from __future__ import annotations

import datetime as dt
import html
import math

import pandas as pd

# --- status / ink tokens (see dataviz skill reference palette) -------------
GOOD = "#0ca30c"
GOOD_DARK = "#0ca30c"
WARNING = "#fab219"
MUTED = "#898781"
SERIES_BLUE_LIGHT = "#2a78d6"
SERIES_BLUE_DARK = "#3987e5"


def _fmt(value, kind: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "&mdash;"
    try:
        if kind == "usd_b":
            # value is a raw dollar amount (e.g. yfinance marketCap)
            if value >= 1e12:
                return f"${value/1e12:,.2f}T"
            if value >= 1e9:
                return f"${value/1e9:,.1f}B"
            if value >= 1e6:
                return f"${value/1e6:,.1f}M"
            return f"${value:,.0f}"
        if kind == "usd":
            return f"${value:,.2f}"
        if kind == "x":
            return f"{value:.1f}x"
        if kind == "pct":
            return f"{value*100:,.1f}%"
        if kind == "pct0":
            return f"{value*100:,.0f}%"
        if kind == "int":
            return f"{value:,.0f}"
        if kind == "score":
            return f"{value:,.0f}"
        return html.escape(str(value))
    except (TypeError, ValueError):
        return "&mdash;"


def _badge(is_true, true_label, false_label="&mdash;") -> str:
    if is_true is None or (isinstance(is_true, float) and math.isnan(is_true)):
        return f'<span class="badge badge-muted">n/a</span>'
    if is_true:
        return f'<span class="badge badge-good">{true_label}</span>'
    return f'<span class="badge badge-muted">{false_label}</span>'


def _score_bar(score) -> str:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "&mdash;"
    pct = max(0.0, min(100.0, float(score)))
    return (
        f'<div class="scorewrap" title="{pct:.1f} / 100">'
        f'<div class="scorebar" style="width:{pct:.1f}%"></div>'
        f'<span class="scoretext">{pct:.0f}</span>'
        f"</div>"
    )


ROW_TEMPLATE = """
<tr>
  <td class="mono">{ticker}</td>
  <td>{company}<div class="muted small">{sector}</div></td>
  <td class="num">{market_cap}</td>
  <td class="num">{price}</td>
  <td class="num">{trailing_pe}</td>
  <td class="num">{forward_pe}</td>
  <td class="num">{peg}</td>
  <td class="num">{debt_equity}</td>
  <td class="num">{roe}</td>
  <td class="num">{drawdown}</td>
  <td>{pullback_badge}</td>
  <td class="num">{vol_ratio}</td>
  <td>{volume_badge}</td>
  <td>{short_badge}</td>
  <td>{score_bar}</td>
</tr>
"""

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Undervalued Stock Scanner &mdash; Results</title>
<style>
  :root {{
    color-scheme: light;
    --surface: #fcfcfb;
    --plane: #f9f9f7;
    --ink: #0b0b0b;
    --ink2: #52514e;
    --muted: #898781;
    --grid: #e1e0d9;
    --border: rgba(11,11,11,0.10);
    --series: {series_light};
    --good: {good};
    --warn: {warn};
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) {{
      color-scheme: dark;
      --surface: #1a1a19;
      --plane: #0d0d0d;
      --ink: #ffffff;
      --ink2: #c3c2b7;
      --muted: #898781;
      --grid: #2c2c2a;
      --border: rgba(255,255,255,0.10);
      --series: {series_dark};
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --surface: #1a1a19;
    --plane: #0d0d0d;
    --ink: #ffffff;
    --ink2: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --border: rgba(255,255,255,0.10);
    --series: {series_dark};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--plane);
    color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 64px; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .subtitle {{ color: var(--ink2); font-size: 13px; margin: 0 0 24px; }}
  .banner {{
    background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--warn);
    border-radius: 8px; padding: 12px 16px; font-size: 13px; color: var(--ink2); margin-bottom: 24px;
  }}
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .tile {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
  .tile .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }}
  .tile .value {{ font-size: 24px; font-weight: 600; margin-top: 4px; font-variant-numeric: tabular-nums; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--surface); border-radius: 10px; overflow: hidden; border: 1px solid var(--border); }}
  thead th {{
    text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: .03em;
    color: var(--muted); padding: 10px 12px; border-bottom: 1px solid var(--grid);
    cursor: pointer; user-select: none; white-space: nowrap; position: sticky; top: 0; background: var(--surface);
  }}
  thead th:hover {{ color: var(--ink); }}
  thead th.sorted::after {{ content: " \\25BE"; }}
  tbody td {{ padding: 9px 12px; border-bottom: 1px solid var(--grid); font-size: 13px; vertical-align: middle; }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover {{ background: var(--plane); }}
  td.num {{ font-variant-numeric: tabular-nums; text-align: right; }}
  td.mono {{ font-variant-numeric: tabular-nums; font-weight: 600; }}
  .muted {{ color: var(--muted); }}
  .small {{ font-size: 11px; }}
  .badge {{
    display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 999px;
  }}
  .badge-good {{ background: color-mix(in srgb, var(--good) 16%, transparent); color: var(--good); }}
  .badge-muted {{ background: color-mix(in srgb, var(--muted) 14%, transparent); color: var(--muted); }}
  .scorewrap {{ position: relative; width: 90px; height: 16px; background: var(--grid); border-radius: 999px; overflow: hidden; }}
  .scorebar {{ position: absolute; inset: 0 auto 0 0; background: var(--series); border-radius: 999px; }}
  .scoretext {{
    position: absolute; inset: 0; display: flex; align-items: center; justify-content: flex-end;
    padding-right: 6px; font-size: 10px; font-weight: 700; font-variant-numeric: tabular-nums; color: var(--ink);
  }}
  footer {{ margin-top: 24px; font-size: 12px; color: var(--muted); }}
  footer a {{ color: var(--ink2); }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Undervalued Stock Scanner</h1>
  <p class="subtitle">Generated {generated_at} &middot; universe: {universe} &middot; {n_scanned} tickers scanned</p>

  <div class="banner">{banner}</div>

  <div class="tiles">
    <div class="tile"><div class="label">Tickers scanned</div><div class="value">{n_scanned}</div></div>
    <div class="tile"><div class="label">Shown below</div><div class="value">{n_shown}</div></div>
    <div class="tile"><div class="label">Low-volume flagged</div><div class="value">{n_low_vol}</div></div>
    <div class="tile"><div class="label">10&ndash;20% pullback flagged</div><div class="value">{n_pullback}</div></div>
    <div class="tile"><div class="label">Median composite score</div><div class="value">{median_score}</div></div>
  </div>

  <table id="results">
    <thead>
      <tr>
        <th data-type="text">Ticker</th>
        <th data-type="text">Company / Sector</th>
        <th data-type="num">Mkt Cap</th>
        <th data-type="num">Price</th>
        <th data-type="num">Trail P/E</th>
        <th data-type="num">Fwd P/E</th>
        <th data-type="num">PEG</th>
        <th data-type="num">Debt/Eq</th>
        <th data-type="num">ROE</th>
        <th data-type="num">Off High</th>
        <th data-type="text">Pullback 10&ndash;20%</th>
        <th data-type="num">Vol Ratio</th>
        <th data-type="text">Low Volume</th>
        <th data-type="text">High Short Int</th>
        <th data-type="num" class="sorted">Score</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>

  <footer>
    Composite score weights your 4 criteria (P/E, historical cheapness, low recent volume, 10&ndash;20% pullback) at 2x,
    and 6 secondary factors (PEG, P/E trend, FCF yield, debt/equity, ROE, P/E vs. sector) at 1x.
    Insider activity and "why is it cheap" are flagged for manual review, not scored &mdash; see the repo README.
    Not investment advice.
  </footer>
</div>
<script>
(function() {{
  const table = document.getElementById('results');
  const tbody = table.tBodies[0];
  const headers = table.tHead.rows[0].cells;
  let sortState = {{ col: 14, dir: -1 }};

  function cellValue(row, idx, type) {{
    const cell = row.cells[idx];
    if (type === 'num') {{
      const t = cell.getAttribute('data-sort') ?? cell.textContent;
      const n = parseFloat(String(t).replace(/[^0-9.\\-]/g, ''));
      return isNaN(n) ? -Infinity : n;
    }}
    return cell.textContent.trim().toLowerCase();
  }}

  function sortBy(idx) {{
    const type = headers[idx].getAttribute('data-type');
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
    Array.from(headers).forEach(h => h.classList.remove('sorted'));
    headers[idx].classList.add('sorted');
  }}

  Array.from(headers).forEach((h, idx) => h.addEventListener('click', () => sortBy(idx)));
}})();
</script>
</body>
</html>
"""


def build_html_report(df: pd.DataFrame, universe_label: str, is_demo: bool, n_total_universe: int | None = None) -> str:
    rows_html = []
    for _, r in df.iterrows():
        rows_html.append(
            ROW_TEMPLATE.format(
                ticker=html.escape(str(r.get("ticker", ""))),
                company=html.escape(str(r.get("company") or "")),
                sector=html.escape(str(r.get("sector") or "")),
                market_cap=_fmt(r.get("market_cap"), "usd_b"),
                price=_fmt(r.get("price"), "usd"),
                trailing_pe=_fmt(r.get("trailing_pe"), "x"),
                forward_pe=_fmt(r.get("forward_pe"), "x"),
                peg=_fmt(r.get("peg"), "x"),
                debt_equity=_fmt(r.get("debt_equity"), "x"),
                roe=_fmt(r.get("roe"), "pct0"),
                drawdown=_fmt(r.get("drawdown_from_high"), "pct0"),
                pullback_badge=_badge(r.get("pullback_10_20_flag"), "in band"),
                vol_ratio=f"{r.get('volume_ratio'):.2f}" if pd.notna(r.get("volume_ratio")) else "&mdash;",
                volume_badge=_badge(r.get("low_volume_flag"), "quiet"),
                short_badge=_badge(r.get("high_short_interest_caution"), "caution"),
                score_bar=_score_bar(r.get("composite_score")),
            )
        )

    n_shown = len(df)
    n_scanned = n_total_universe if n_total_universe is not None else n_shown
    n_low_vol = int(df.get("low_volume_flag", pd.Series(dtype=bool)).fillna(False).sum())
    n_pullback = int(df.get("pullback_10_20_flag", pd.Series(dtype=bool)).fillna(False).sum())
    median_score = df["composite_score"].median() if "composite_score" in df.columns and len(df) else float("nan")

    banner = (
        "DEMO DATA &mdash; these 16 rows are hand-verified sample tickers used to sanity-check the "
        "scoring logic offline. Run <code>python scanner.py --universe both -o results.html</code> "
        "with real internet access to replace this with a live S&amp;P 500 + Nasdaq Composite scan."
        if is_demo else
        "Live scan output. Composite score is a screening aid, not a recommendation &mdash; "
        "see the README's \"10 things to consider\" before acting on anything here."
    )

    return PAGE_TEMPLATE.format(
        series_light=SERIES_BLUE_LIGHT,
        series_dark=SERIES_BLUE_DARK,
        good=GOOD,
        warn=WARNING,
        generated_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        universe=html.escape(universe_label),
        n_scanned=n_scanned,
        n_shown=n_shown,
        n_low_vol=n_low_vol,
        n_pullback=n_pullback,
        median_score=f"{median_score:.0f}" if not math.isnan(median_score) else "&mdash;",
        banner=banner,
        rows="\n".join(rows_html),
    )


def write_html_report(df: pd.DataFrame, path: str, universe_label: str, is_demo: bool, n_total_universe: int | None = None) -> None:
    html_str = build_html_report(df, universe_label, is_demo, n_total_universe)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_str)
