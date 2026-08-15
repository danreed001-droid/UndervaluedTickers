"""
html_report.py
===============
Renders scanner results as a single self-contained, dependency-free HTML
file (no external CSS/JS/fonts) with a click-to-sort table, a per-ticker
120-day price sparkline, status badges, and a composite-score bar. Colors
follow Anthropic's dataviz status/categorical/ink palette (see the dataviz
skill's reference palette) so it reads consistently -- and stays
colorblind-safe -- in both light and dark mode.

Kept separate from scanner.py so the scoring logic and the presentation
layer don't get tangled.
"""

from __future__ import annotations

import datetime as dt
import html
import math

import pandas as pd

# --- status / ink tokens (see dataviz skill reference palette) -------------
# These are mode-invariant per the skill's status palette -- same hex in
# light and dark, they just land at different contrast against each surface.
GOOD = "#0ca30c"       # positive / desirable flag (e.g. quiet volume, healthy pullback, price up)
WARNING = "#fab219"    # caution (e.g. high short interest)
CRITICAL = "#d03b3b"   # negative direction (e.g. price down over the window)
MUTED = "#898781"      # neutral / not-flagged / axis text

# Categorical slot 1 (blue) -- used for the primary "this is the headline
# metric" accent (composite score, "shown" tile). Single-hue use only, so no
# validator re-run is needed beyond the reference instance.
SERIES_BLUE_LIGHT = "#2a78d6"
SERIES_BLUE_DARK = "#3987e5"
# Categorical slot 3 (aqua) -- a second, distinct accent for a different tile
# so the KPI row isn't monochrome. Not used to encode identity across rows,
# so it doesn't need to clear the all-pairs gate.
SERIES_AQUA_LIGHT = "#1baf7a"
SERIES_AQUA_DARK = "#199e70"


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


def _badge(is_true, true_label, false_label="&mdash;", variant="good") -> str:
    """variant is 'good' or 'warning' -- picks which status color the true-state gets."""
    if is_true is None or (isinstance(is_true, float) and math.isnan(is_true)):
        return '<span class="badge badge-muted">n/a</span>'
    if is_true:
        cls = "badge-warning" if variant == "warning" else "badge-good"
        return f'<span class="badge {cls}">{true_label}</span>'
    return f'<span class="badge badge-muted">{false_label}</span>'


def _sector_chips_html(df: pd.DataFrame) -> str:
    """Clickable sector filter chips above the table. Built from whatever
    sectors are actually present in this run's data (not a hardcoded GICS
    list) so a chip never points at zero rows. 'All' clears the filter."""
    if "sector" not in df.columns:
        return ""
    sectors = sorted(s for s in df["sector"].dropna().unique().tolist() if str(s).strip())
    if not sectors:
        return ""
    chips = ['<button type="button" class="chip chip-all active" data-sector="">All sectors</button>']
    for s in sectors:
        s_esc = html.escape(str(s))
        chips.append(f'<button type="button" class="chip" data-sector="{s_esc}">{s_esc}</button>')
    return "\n    ".join(chips)


def _score_tier(score: float) -> str:
    """Composite score -> CSS modifier class. Three tiers, status-colored."""
    if score >= 75:
        return "tier-good"
    if score >= 50:
        return "tier-warning"
    return "tier-muted"


def _score_bar(score) -> str:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "&mdash;"
    pct = max(0.0, min(100.0, float(score)))
    tier = _score_tier(pct)
    return (
        f'<div class="scorewrap {tier}" title="{pct:.1f} / 100">'
        f'<div class="scorebar" style="width:{pct:.1f}%"></div>'
        f'<span class="scoretext">{pct:.0f}</span>'
        f"</div>"
    )


def _parse_sparkline(raw) -> list[float]:
    """sparkline_prices is stored as a ';'-joined string so it survives the
    scanner's CSV checkpoint round-trip; parse it back to floats here."""
    if raw is None:
        return []
    if isinstance(raw, float) and math.isnan(raw):
        return []
    if isinstance(raw, (list, tuple)):
        vals = raw
    else:
        s = str(raw).strip()
        if not s:
            return []
        vals = s.split(";")
    out = []
    for v in vals:
        try:
            f = float(v)
            if not math.isnan(f):
                out.append(f)
        except (TypeError, ValueError):
            continue
    return out


def _sparkline_svg(prices: list[float], width: int = 96, height: int = 26) -> str:
    """A compact 120-day price trend line: thin 2px line + faint area wash,
    colored by direction (status good = up, status critical = down), with an
    end-dot and a signed % delta so direction never rides on hue alone."""
    if len(prices) < 2:
        return '<span class="muted small">&mdash;</span>'

    lo, hi = min(prices), max(prices)
    rng = (hi - lo) or 1.0
    n = len(prices)
    pad_y = 3.0
    step = width / (n - 1)

    def y(v: float) -> float:
        return pad_y + (hi - v) / rng * (height - 2 * pad_y)

    pts = [(i * step, y(v)) for i, v in enumerate(prices)]
    line_d = "M " + " L ".join(f"{x:.1f},{yy:.1f}" for x, yy in pts)
    baseline = height - pad_y
    area_d = (
        f"M {pts[0][0]:.1f},{baseline:.1f} "
        + " L ".join(f"{x:.1f},{yy:.1f}" for x, yy in pts)
        + f" L {pts[-1][0]:.1f},{baseline:.1f} Z"
    )

    up = prices[-1] >= prices[0]
    color_var = "var(--good)" if up else "var(--critical)"
    pct_change = (prices[-1] / prices[0] - 1.0) * 100 if prices[0] else 0.0
    end_x, end_y = pts[-1]
    title = f"Last {n} sessions: {prices[0]:.2f} → {prices[-1]:.2f} ({pct_change:+.1f}%)"

    svg = (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'class="spark" role="img" aria-label="{html.escape(title)}">'
        f"<title>{html.escape(title)}</title>"
        f'<path d="{area_d}" fill="{color_var}" fill-opacity="0.12" stroke="none"/>'
        f'<path d="{line_d}" fill="none" stroke="{color_var}" stroke-width="1.75" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="2.5" fill="{color_var}" '
        f'stroke="var(--surface)" stroke-width="1.4"/>'
        f"</svg>"
    )
    delta_cls = "delta-good" if up else "delta-bad"
    delta_html = f'<span class="delta {delta_cls}">{pct_change:+.1f}%</span>'
    return f'<div class="sparkwrap">{svg}{delta_html}</div>'


ROW_TEMPLATE = """
<tr data-sector="{sector_attr}">
  <td class="mono col-rank">{rank}</td>
  <td class="mono">{ticker}</td>
  <td>{company}<div class="muted small">{sector}</div></td>
  <td>{sparkline}</td>
  <td>{score_bar}</td>
  <td class="blurb">{blurb}</td>
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
    --series2: {series2_light};
    --good: {good};
    --warn: {warn};
    --critical: {critical};
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
      --series2: {series2_dark};
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
    --series2: {series2_dark};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--plane);
    color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 1280px; margin: 0 auto; padding: 32px 20px 64px; }}
  h1 {{ font-size: 24px; margin: 0 0 4px; letter-spacing: -0.01em; }}
  .accent {{ color: var(--series); }}
  .subtitle {{ color: var(--ink2); font-size: 13px; margin: 0 0 24px; }}
  .banner {{
    background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--warn);
    border-radius: 8px; padding: 12px 16px; font-size: 13px; color: var(--ink2); margin-bottom: 24px;
  }}
  .banner-note {{
    margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border);
    color: var(--critical); font-weight: 600;
  }}
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .tile {{
    background: var(--surface); border: 1px solid var(--border); border-top: 3px solid var(--tile-accent, var(--grid));
    border-radius: 10px; padding: 14px 16px;
  }}
  .tile.tile-blue {{ --tile-accent: var(--series); }}
  .tile.tile-aqua {{ --tile-accent: var(--series2); }}
  .tile.tile-good {{ --tile-accent: var(--good); }}
  .tile .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }}
  .tile .value {{ font-size: 26px; font-weight: 600; margin-top: 4px; }}

  .filterbar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 12px; }}
  .chip {{
    font: inherit; font-size: 12px; font-weight: 600; padding: 6px 14px; border-radius: 999px;
    border: 1px solid var(--border); background: var(--surface); color: var(--ink2);
    cursor: pointer; white-space: nowrap;
  }}
  .chip:hover {{ background: var(--plane); }}
  .chip.active {{ background: var(--series); border-color: var(--series); color: #fff; }}
  .filter-count {{ font-size: 12px; color: var(--muted); margin-left: auto; white-space: nowrap; }}

  /* One bounded, both-axis scroll panel for the table -- the tiles, filter
     bar and headline stay in normal page flow above it. This is required
     (not just nicer) for the sticky header + sticky rank column below:
     position: sticky needs a real scrolling ancestor, and it has to be the
     SAME ancestor for both the thead (top: 0) and .col-rank (left: 0). An
     unbounded-height wrapper with only overflow-x set doesn't work for the
     vertical stick (there'd be nothing to scroll internally); a bounded
     max-height with overflow: auto on both axes gives both their anchor. */
  .table-scroll {{ max-height: 74vh; overflow: auto; border-radius: 10px; border: 1px solid var(--border); }}
  table {{ width: 100%; border-collapse: collapse; background: var(--surface); }}
  thead th {{
    text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: .03em;
    color: var(--muted); padding: 10px 12px; border-bottom: 1px solid var(--grid);
    cursor: pointer; user-select: none; white-space: nowrap; position: sticky; top: 0; background: var(--surface);
  }}
  thead th:hover {{ color: var(--ink); }}
  thead th.sorted::after {{ content: " \\25BE"; }}
  thead th.nosort {{ cursor: default; }}
  thead th.nosort:hover {{ color: var(--muted); }}
  tbody td {{ padding: 9px 12px; border-bottom: 1px solid var(--grid); font-size: 13px; vertical-align: middle; }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover {{ background: var(--plane); }}
  tbody tr:hover td.col-rank {{ background: var(--plane); }}
  td.num {{ font-variant-numeric: tabular-nums; text-align: right; }}
  td.mono {{ font-variant-numeric: tabular-nums; font-weight: 600; }}
  /* Rank ("#") stays pinned to the left edge -- on a small screen this
     table is both long (scroll down) and wide (scroll sideways), and this
     is the one column that answers "which row am I even looking at" in
     either direction. */
  .col-rank {{
    position: sticky; left: 0; z-index: 1; background: var(--surface);
    text-align: center; color: var(--muted); width: 1%;
  }}
  thead th.col-rank {{ z-index: 3; }}
  td.blurb {{ font-size: 12px; color: var(--ink2); max-width: 260px; min-width: 200px; white-space: normal; line-height: 1.4; }}
  .muted {{ color: var(--muted); }}
  .small {{ font-size: 11px; }}
  .badge {{
    display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 999px; white-space: nowrap;
  }}
  .badge-good {{ background: color-mix(in srgb, var(--good) 16%, transparent); color: var(--good); }}
  .badge-warning {{ background: color-mix(in srgb, var(--warn) 22%, transparent); color: color-mix(in srgb, var(--warn) 70%, var(--ink)); }}
  .badge-muted {{ background: color-mix(in srgb, var(--muted) 14%, transparent); color: var(--muted); }}

  .sparkwrap {{ display: flex; align-items: center; gap: 6px; }}
  .spark {{ display: block; flex: none; }}
  .delta {{ font-size: 11px; font-weight: 700; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  .delta-good {{ color: var(--good); }}
  .delta-bad {{ color: var(--critical); }}

  .scorewrap {{ position: relative; width: 90px; height: 16px; background: var(--grid); border-radius: 999px; overflow: hidden; }}
  .scorebar {{ position: absolute; inset: 0 auto 0 0; background: var(--tier-color, var(--series)); border-radius: 999px; }}
  .scorewrap.tier-good {{ --tier-color: var(--good); }}
  .scorewrap.tier-warning {{ --tier-color: var(--warn); }}
  .scorewrap.tier-muted {{ --tier-color: var(--muted); }}
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
  <h1><span class="accent">&#9679;</span> Undervalued Stock Scanner</h1>
  <p class="subtitle">Generated {generated_at} &middot; universe: {universe} &middot; {n_scanned} tickers scanned</p>

  <div class="banner">{banner}</div>

  <div class="tiles">
    <div class="tile"><div class="label">Tickers scanned</div><div class="value">{n_scanned}</div></div>
    <div class="tile tile-blue"><div class="label">Shown below</div><div class="value">{n_shown}</div></div>
    <div class="tile tile-good"><div class="label">Low-volume flagged</div><div class="value">{n_low_vol}</div></div>
    <div class="tile tile-good"><div class="label">10&ndash;20% pullback flagged</div><div class="value">{n_pullback}</div></div>
    <div class="tile tile-blue"><div class="label">Median composite score</div><div class="value">{median_score}</div></div>
  </div>

  <div class="filterbar" id="sector-filter">
    {sector_chips}
    <span class="filter-count" id="filter-count">Showing {n_shown} of {n_shown}</span>
  </div>

  <div class="table-scroll">
  <table id="results">
    <thead>
      <tr>
        <th data-type="skip" class="nosort col-rank">#</th>
        <th data-type="text">Ticker</th>
        <th data-type="text">Company / Sector</th>
        <th data-type="skip" class="nosort">Trend (120d)</th>
        <th data-type="num" class="sorted">Score</th>
        <th data-type="text">Why It Scored This Way</th>
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
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  </div>

  <footer>
    Composite score weights your 4 criteria (P/E, historical cheapness, low recent volume, 10&ndash;20% pullback) at 2x,
    and 6 secondary factors (PEG, P/E trend, FCF yield, debt/equity, ROE, P/E vs. sector) at 1x. Score bar color is a tier
    (75+ / 50&ndash;74 / below 50), not a ranking. The Trend sparkline is each ticker's last ~120 trading-day close, colored
    by direction, independent of the composite score. Insider activity and "why is it cheap" are flagged for manual
    review, not scored &mdash; see the repo README. Not investment advice.
  </footer>
</div>
<script>
(function() {{
  const table = document.getElementById('results');
  const tbody = table.tBodies[0];
  const headers = table.tHead.rows[0].cells;
  let sortState = {{ col: 4, dir: -1 }};

  // The "#" column is a running position among whatever rows are currently
  // visible, in whatever order is currently on screen -- not a fixed
  // composite-score rank. Recomputed after every sort AND every filter
  // change so it never goes stale or counts hidden rows.
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
      const t = cell.getAttribute('data-sort') ?? cell.textContent;
      const n = parseFloat(String(t).replace(/[^0-9.\\-]/g, ''));
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

  // Sector filter chips -- click one to show only that sector, click it
  // again (or click "All sectors") to clear back to everything. Built
  // dynamically server-side from whatever sectors are actually in this
  // report, so there's no chip that ever points at zero rows.
  const filterbar = document.getElementById('sector-filter');
  if (filterbar) {{
    const chips = Array.from(filterbar.querySelectorAll('.chip'));
    const allChip = filterbar.querySelector('.chip-all');
    const countEl = document.getElementById('filter-count');
    const rows = Array.from(tbody.rows);
    const total = rows.length;

    function applyFilter(sector) {{
      rows.forEach(r => {{
        const match = !sector || r.getAttribute('data-sector') === sector;
        r.style.display = match ? '' : 'none';
      }});
      renumberVisibleRanks();
      if (countEl) {{
        const shown = rows.filter(r => r.style.display !== 'none').length;
        countEl.textContent = sector ? `Showing ${{shown}} of ${{total}} — ${{sector}}` : `Showing ${{total}} of ${{total}}`;
      }}
    }}

    chips.forEach(chip => {{
      chip.addEventListener('click', () => {{
        const sector = chip.getAttribute('data-sector');
        const alreadyActive = chip.classList.contains('active');
        chips.forEach(c => c.classList.remove('active'));
        if (!sector || alreadyActive) {{
          if (allChip) allChip.classList.add('active');
          applyFilter('');
        }} else {{
          chip.classList.add('active');
          applyFilter(sector);
        }}
      }});
    }});
  }}
}})();
</script>
</body>
</html>
"""


def build_html_report(df: pd.DataFrame, universe_label: str, is_demo: bool, n_total_universe: int | None = None,
                       universe_note: str | None = None) -> str:
    rows_html = []
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        sector_val = html.escape(str(r.get("sector") or ""))
        rows_html.append(
            ROW_TEMPLATE.format(
                rank=i,
                sector_attr=sector_val,
                ticker=html.escape(str(r.get("ticker", ""))),
                company=html.escape(str(r.get("company") or "")),
                sector=sector_val,
                sparkline=_sparkline_svg(_parse_sparkline(r.get("sparkline_prices"))),
                blurb=html.escape(str(r.get("score_blurb") or "")),
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
                short_badge=_badge(r.get("high_short_interest_caution"), "caution", variant="warning"),
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
        "scoring logic offline (with a synthetic 120-day trend line, since demo mode makes no network "
        "calls). Run <code>python scanner.py --universe both -o results.html</code> "
        "with real internet access to replace this with a live S&amp;P 500 + Nasdaq Composite scan "
        "and real price history."
        if is_demo else
        "Live scan output. Composite score is a screening aid, not a recommendation &mdash; "
        "see the README's \"10 things to consider\" before acting on anything here."
    )
    if universe_note:
        banner += (
            f'<div class="banner-note">&#9888; {html.escape(universe_note)}</div>'
        )

    return PAGE_TEMPLATE.format(
        series_light=SERIES_BLUE_LIGHT,
        series_dark=SERIES_BLUE_DARK,
        series2_light=SERIES_AQUA_LIGHT,
        series2_dark=SERIES_AQUA_DARK,
        good=GOOD,
        warn=WARNING,
        critical=CRITICAL,
        generated_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        universe=html.escape(universe_label),
        n_scanned=n_scanned,
        n_shown=n_shown,
        n_low_vol=n_low_vol,
        n_pullback=n_pullback,
        median_score=f"{median_score:.0f}" if not math.isnan(median_score) else "&mdash;",
        banner=banner,
        sector_chips=_sector_chips_html(df),
        rows="\n".join(rows_html),
    )


def write_html_report(df: pd.DataFrame, path: str, universe_label: str, is_demo: bool, n_total_universe: int | None = None,
                       universe_note: str | None = None) -> None:
    html_str = build_html_report(df, universe_label, is_demo, n_total_universe, universe_note)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_str)
