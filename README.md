# Undervalued Stock Scanner

A P/E + historical-valuation + volume + pullback screener for the S&P 500
and Nasdaq Composite, with a transparent composite score and an HTML
report you can open in a browser.

## What it screens for

Four criteria, weighted **2x** each in the composite score:

1. **P/E ratio** — cheap relative to the rest of the scanned universe / its sector
2. **Historical valuation** — current P/E cheap relative to the stock's own ~5-year norm
3. **Low recent volume** — average volume over the last 10–20 trading days running below its normal (1-year) average
4. **Pullback** — price currently 10–20% below its 52-week high

Six more factors, weighted **1x** each (drawn from a "10 things to
consider beyond P/E and volume" list):

- PEG ratio
- Trailing vs. forward P/E trend (is the market pricing in earnings growth?)
- Free cash flow yield
- Debt/equity
- Return on equity
- P/E vs. sector peers

Dividend payout sustainability is also scored (1x) where a stock pays a dividend.

Two more factors from that same 10-point list — **insider buying/selling**
and **why the market is pricing the stock low** — are *not* scored. There's
no reliable free bulk data source for insider activity, and "why is it
cheap" is inherently qualitative. Every row in the report is flagged
"review manually" for both before you act on it.

All scoring is **self-relative**: a stock's P/E, for example, is graded
against the quantile distribution of everything else scanned in the same
run (and against its own sector for the sector comparison), not an
arbitrary fixed cutoff. That keeps it sensible across market regimes
instead of hardcoding "P/E under 15 = cheap" forever.

## Quick start

```bash
pip install -r requirements.txt

# Sanity-check the scoring logic with built-in sample data, no network needed:
python scanner.py --demo -o results.html

# Real run against 25 random tickers first, to make sure yfinance behaves
# on your machine before committing to a multi-thousand-ticker scan:
python scanner.py --universe both --limit 25 -o results.html

# Full scan: S&P 500 + Nasdaq Composite (~3,500+ tickers). This is several
# thousand network calls and can take 30-90+ minutes. It checkpoints to
# CSV as it runs, so Ctrl-C is safe -- rerun with --resume to continue.
python scanner.py --universe both --workers 20 -o results.html
python scanner.py --universe both --resume -o results.html   # if interrupted
```

Open `results.html` in a browser — it's a single self-contained file
(no external CSS/JS), with a click-to-sort table and light/dark mode.

`-o` also accepts `.xlsx` or `.csv` if you'd rather work in a spreadsheet.

## Files

| File | What it does |
|---|---|
| `scanner.py` | Universe building (S&P 500 via Wikipedia, Nasdaq Composite via nasdaqtrader.com), threaded fetch from yfinance, vectorized scoring, CLI |
| `html_report.py` | Renders the scored results as a single-file HTML report |
| `requirements.txt` | Python dependencies |

## Important caveats

- **Needs real internet access.** `yfinance` pulls live data from Yahoo
  Finance; this will not work in a network-sandboxed environment.
- **Nasdaq Composite is approximated**, not an official index feed — it's
  built from Nasdaq's public listed-securities file, filtered to common
  shares. The real index includes a handful of edge cases (some non-U.S.
  listed ADRs, etc.) this doesn't capture perfectly.
- **This is a screening aid, not investment advice.** A high composite
  score means a stock looks statistically cheap and quiet/pulled-back
  by the metrics available — it says nothing about *why*, which is the
  one thing that matters most. Read the "review manually" flag on every
  row before acting on it.

## Example output

`results.html` in this repo is a committed example — 16 hand-verified
sample tickers (generated with `--demo`), so you can see the report format
without running a live scan first.
