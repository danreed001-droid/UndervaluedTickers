"""Shared ticker universes / timeframe config for the dual-timeframe VSA
scan (scripts/dual_timeframe_scan.py). Kept as its own module -- separate
from scan_volume.py's own S&P 500 / Nasdaq Composite / fixed-ETF universe --
since this is a different, curated watchlist keyed by theme.
"""
from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
#  ① UNIVERSE / CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SP500 = [
    "A","AAL","AAPL","ABBV","ABNB","ABT","ACGL","ACN","ADBE","ADI","ADM","ADSK",
    "AEE","AEP","AES","AFL","AIG","AIZ","AJG","AKAM","ALB","ALGN","ALL","ALLE",
    "AMAT","AMCR","AMD","AME","AMGN","AMP","AMZN","ANET","ANSS","AON","AOS","APA",
    "APD","APH","APO","APTV","ARE","ATO","AVB","AVGO","AVY","AWK","AXON","AXP",
    "AZO","BA","BAC","BALL","BAX","BBWI","BBY","BDX","BEN","BF-B","BG","BIIB",
    "BK","BKNG","BLK","BLDR","BMY","BR","BRK-B","BSX","BWA","BX","BYD","C",
    "CARR","CAT","CB","CBOE","CBRE","CCI","CDNS","CDW","CE","CEG","CF","CFG",
    "CHD","CHTR","CI","CINF","CL","CLX","CMA","CMCSA","CME","CMG","CMI","CMS",
    "CNC","CNP","COF","COO","COP","COST","CPAY","CPB","CPRT","CPT","CRL","CRM",
    "CSGP","CSX","CTAS","CTRA","CTSH","CTVA","CVS","CVX","CZR","D","DASH","DAY",
    "DD","DE","DECK","DELL","DFS","DG","DGX","DHI","DHR","DIS","DLR","DLTR",
    "DOV","DOW","DPZ","DRI","DTE","DUK","DVA","DVN","DXCM","EA","EBAY","ECL",
    "ED","EFX","EG","EIX","EL","ELV","EMN","EMR","ENPH","EOG","EPAM","EQIX",
    "EQR","EQT","ES","ESS","ETN","ETR","ETSY","EVRG","EW","EXC","EXPD","EXPE",
    "EXR","F","FANG","FAST","FCX","FDS","FDX","FE","FI","FICO","FIS","FITB",
    "FMC","FOX","FOXA","FRT","FTNT","FTV","GD","GE","GEHC","GEN","GEV","GILD",
    "GIS","GL","GLW","GM","GNRC","GOOG","GOOGL","GPN","GRMN","GS","GWW","HAL",
    "HAS","HCA","HD","HES","HIG","HII","HLT","HOLX","HON","HPE","HPQ","HRL",
    "HSIC","HST","HSY","HUBB","HUM","HWM","IBM","ICE","IDXX","IEX","IFF","ILMN",
    "INCY","INTC","INTU","INVH","IP","IPG","IQV","IR","IRM","ISRG","IT","ITW",
    "IVZ","J","JBHT","JBL","JCI","JKHY","JNJ","JNPR","JPM","K","KDP","KEY",
    "KEYS","KHC","KIM","KLAC","KMB","KMI","KMX","KO","KR","KVUE","L","LDOS",
    "LEN","LH","LHX","LIN","LKQ","LLY","LMT","LNC","LNT","LOW","LRCX","LULU",
    "LUV","LVS","LW","LYB","LYV","MA","MAA","MAR","MAS","MCK","MCO","MDLZ",
    "MDT","MET","META","MGM","MHK","MKC","MKTX","MLM","MMC","MMM","MNST","MO",
    "MOH","MOS","MPC","MPWR","MRK","MRNA","MS","MSCI","MSFT","MSI","MTB","MTCH",
    "MTD","MU","NCLH","NDAQ","NDSN","NEE","NEM","NFLX","NI","NKE","NOC","NOW",
    "NRG","NSC","NTAP","NTRS","NUE","NVDA","NVR","NWS","NWSA","NXPI","O","ODFL",
    "OKE","OMC","ON","ORCL","ORLY","OTIS","OXY","PANW","PARA","PAYC","PAYX",
    "PCG","PEG","PEP","PFE","PFG","PG","PGR","PH","PHM","PKG","PLD","PLTR",
    "PM","PNC","PNW","PODD","POOL","PPG","PPL","PRU","PSA","PSX","PTC","PWR",
    "PYPL","QCOM","QRVO","RCL","REG","REGN","RF","RHI","RJF","RL","RMD","ROK",
    "ROL","ROP","ROST","RSG","RTX","RVTY","SBAC","SBUX","SCHW","SHW","SJM",
    "SLB","SMCI","SNA","SNPS","SO","SPG","SPGI","SRE","STE","STLD","STT","STX",
    "SYK","SYF","SYY","T","TAP","TDG","TDY","TECH","TEL","TER","TFC","TFX",
    "TGT","TJX","TMO","TMUS","TPR","TRGP","TRMB","TROW","TRV","TSCO","TSLA",
    "TSN","TT","TTD","TTWO","TXN","TXT","TYL","UAL","UBER","UDR","UHS","ULTA",
    "UNH","UNP","UPS","URI","USB","V","VICI","VLO","VMC","VRSK","VRSN","VRTX",
    "VTR","VTRS","VZ","WAB","WAT","WBA","WBD","WDC","WEC","WELL","WFC","WMB",
    "WMT","WRB","WST","WTW","WY","WYNN","XEL","XOM","XRAY","XYL","YUM","ZBH",
    "ZBRA","ZTS"
]
SC_ETFS   = ["XLC","XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLRE","XLK","XLU","SPY","QQQ","IWM"]
SC_FX     = ["EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X"]
SC_CRYPTO = ["BTC-USD","ETH-USD","SOL-USD"]

TF_ORDER = ["5m", "15m", "1h", "1d"]
TF_MAP   = {
    "5m":  ("30d", "5m"),
    "15m": ("30d", "15m"),
    "1h":  ("1y",  "1h"),
    "1d":  ("max", "1d"),
}

# ══════════════════════════════════════════════════════════════════════════════
#  ② VSA — ASSET DATASET & COLORS
# ══════════════════════════════════════════════════════════════════════════════

VSA_ASSETS = {
    "TECH & AI":        ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AVGO","ORCL","AMD","NFLX","PLTR","SMCI"],
    "FINANCE & VALUE":  ["JPM","BAC","GS","BRK-B","V","MA","COST","WMT","XOM","CVX"],
    "ETFs":             ["SPY","QQQ","IWM","DIA","GLD","TLT","XLK","XLF","SMH","ARKK","XLC","XLY","XLU","XLP","XLRE"],
    "CRYPTO & FX":      ["BTC-USD","ETH-USD","SOL-USD","EURUSD=X","USDJPY=X","GBPUSD=X","AUDUSD=X"],
    "COMMODITIES":      ["GC=F","SI=F","CL=F","NG=F"],
    "INDUSTRIALS":      ["CAT","HON","GE","UNP","UPS","DE","MMM"],
    "CONSUMER STAPLES": ["KO","PEP","PG","COST","MDLZ","PM","TGT","EL"],
    "UTILITIES":        ["NEE","DUK","SO","AEP","SRE","D","EXC"],
    "MATERIALS":        ["LIN","SHW","BHP","NEM","SCCO","FCX","CTVA","ECL"],
    "REAL ESTATE":      ["PLD","AMT","EQIX","DLR","SPG","PSA","WY","AVB"],
    "BIOTECH & PHARMA": ["LLY","NVO","REGN","VRTX","AMGN","BNTX","GILD","AXSM"],
    "AERO & DEFENSE":   ["LMT","RTX","NOC","BA","GD","LHX","AVAV","LDOS"],
    "CYBERSECURITY":    ["CRWD","PANW","FTNT","ZS","OKTA","NET","S"],
    "NUCLEAR & POWER":  ["SMR","OKLO","BWXT","CCJ","UUUU","VST","TLN"],
    "TRAVEL & LUXURY":  ["LVMUY","RACE","HESAY","BKNG","DAL","H","MAR","EXPE"],
}

# ══════════════════════════════════════════════════════════════════════════════
#  ③ UNIFIED FILTER GROUPS -- for the dual-timeframe report's group chips.
#     A ticker can (and often does) belong to more than one group -- e.g.
#     AAPL is in both "S&P 500" and "TECH & AI". Rows are deduplicated by
#     ticker; the chips just narrow which rows are visible.
# ══════════════════════════════════════════════════════════════════════════════

GROUPS: dict[str, list[str]] = {
    "S&P 500": SP500,
    "ETF Watchlist": SC_ETFS,
    "FX": SC_FX,
    "Crypto": SC_CRYPTO,
    **VSA_ASSETS,
}


def ticker_groups_map() -> dict[str, list[str]]:
    """ticker -> sorted list of every group name it appears in."""
    out: dict[str, list[str]] = {}
    for group_name, tickers in GROUPS.items():
        for t in tickers:
            out.setdefault(t, [])
            if group_name not in out[t]:
                out[t].append(group_name)
    return {t: sorted(groups) for t, groups in out.items()}


def all_tickers() -> list[str]:
    return sorted(ticker_groups_map().keys())
