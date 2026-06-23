"""
Data Lake Builder — downloads A-share data from Tushare and builds a DuckDB catalog.

Usage: python3 src/data/build_lake.py

Data stored in lake/catalog.duckdb with tables:
  - stocks_daily: adjusted OHLCV + turnover
  - stocks_basic: stock listing info (ts_code, name, industry, list_date, etc.)
  - stocks_daily_basic: PE, PB, market_cap, dividend_yield
  - adj_factors: cumulative adjustment factors
  - index_daily: index OHLCV
  - index_member: index constituent mapping
  - index_classify: industry classification (88-series)
  - fund_daily: ETF daily prices
  - fund_basic: ETF listing info
  - hk_hold: northbound holdings
  - margin: margin financing data
  - trade_cal: trading calendar
"""

import os
import time
import json
from datetime import datetime, date
from pathlib import Path

import duckdb
import pandas as pd
import tushare as ts

# ── Configuration ──────────────────────────────────────────────────────────

LAKE_DIR = Path(__file__).parent.parent.parent / "lake"
CATALOG_PATH = LAKE_DIR / "catalog.duckdb"
MANIFEST_DIR = LAKE_DIR / "_manifests"

START_DATE = "20050101"  # reliable data from 2005
# Key indices
INDEX_CODES = {
    "000300.SH": "CSI300",
    "000905.SH": "CSI500", 
    "000016.SH": "SSE50",
    "399006.SZ": "ChiNext",
    "000852.SH": "CSI1000",
}

# Rate limiting
CALL_DELAY = 0.3  # seconds between API calls
BATCH_SIZE = 5000  # max rows per Tushare call


def get_pro():
    """Get Tushare Pro API instance."""
    token = os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        # Try to read from default config
        cfg_path = Path.home() / ".tushare" / "token"
        if cfg_path.exists():
            token = cfg_path.read_text().strip()
    pro = ts.pro_api(token) if token else ts.pro_api()
    return pro


def rate_limit(last_call: float, delay: float = CALL_DELAY):
    """Enforce rate limit between API calls."""
    elapsed = time.time() - last_call
    if elapsed < delay:
        time.sleep(delay - elapsed)
    return time.time()


def fetch_all(pro, func_name: str, **kwargs) -> pd.DataFrame:
    """Fetch all pages from a Tushare endpoint with rate limiting."""
    func = getattr(pro, func_name)
    all_data = []
    offset = 0
    last_call = 0.0
    page = 0
    
    while True:
        last_call = rate_limit(last_call)
        try:
            df = func(**kwargs, limit=BATCH_SIZE, offset=offset)
        except Exception as e:
            print(f"  Error at offset {offset}: {e}, retrying in 5s...")
            time.sleep(5)
            last_call = time.time()
            continue
        
        if df is None or len(df) == 0:
            break
        
        all_data.append(df)
        page += 1
        if page % 10 == 0:
            print(f"  ... page {page}, {len(all_data) * BATCH_SIZE:,} rows so far")
        
        if len(df) < BATCH_SIZE:
            break
        
        offset += BATCH_SIZE
    
    if not all_data:
        return pd.DataFrame()
    
    result = pd.concat(all_data, ignore_index=True)
    print(f"  Downloaded {len(result):,} rows from {func_name}")
    return result


# ── Data Download Functions ────────────────────────────────────────────────

def download_stock_basic(pro) -> pd.DataFrame:
    """Download stock listing info."""
    print("\n[stock_basic] Downloading...")
    df = pro.stock_basic(
        exchange="", 
        list_status="L",  # listed only
        fields="ts_code,symbol,name,area,industry,market,list_date,delist_date,curr_type"
    )
    # Also get delisted stocks
    df_d = pro.stock_basic(
        exchange="",
        list_status="D",
        fields="ts_code,symbol,name,area,industry,market,list_date,delist_date,curr_type"
    )
    # Also suspended
    df_p = pro.stock_basic(
        exchange="",
        list_status="P",
        fields="ts_code,symbol,name,area,industry,market,list_date,delist_date,curr_type"
    )
    df = pd.concat([df, df_d, df_p], ignore_index=True)
    print(f"  Downloaded {len(df):,} stock records")
    return df


def download_trade_cal(pro) -> pd.DataFrame:
    """Download trading calendar."""
    print("\n[trade_cal] Downloading...")
    df = pro.trade_cal(exchange="SSE", start_date="20000101", end_date="20991231")
    print(f"  Downloaded {len(df):,} calendar records")
    return df


def download_daily(pro, start: str, end: str) -> pd.DataFrame:
    """Download daily stock OHLCV."""
    print(f"\n[stocks_daily] Downloading {start} → {end}...")
    
    # Download in chunks by year to respect API limits
    years = list(range(int(start[:4]), int(end[:4]) + 1))
    all_data = []
    
    for yr in years:
        s = f"{yr}0101"
        e = f"{yr}1231"
        if yr == int(start[:4]):
            s = start
        if yr == int(end[:4]):
            e = end
        
        print(f"  Year {yr}...")
        df = fetch_all(pro, "daily", ts_code="", start_date=s, end_date=e,
                       fields="ts_code,trade_date,open,high,low,close,pre_close,vol,amount")
        if len(df) > 0:
            all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    result = pd.concat(all_data, ignore_index=True)
    print(f"  Total stocks_daily: {len(result):,} rows")
    return result


def download_adj_factor(pro, start: str, end: str) -> pd.DataFrame:
    """Download adjustment factors."""
    print(f"\n[adj_factor] Downloading...")
    df = fetch_all(pro, "adj_factor", ts_code="", start_date=start, end_date=end,
                   fields="ts_code,trade_date,adj_factor")
    return df


def download_daily_basic(pro, start: str, end: str) -> pd.DataFrame:
    """Download daily basic data (PE, PB, market cap, etc.)."""
    print(f"\n[daily_basic] Downloading...")
    
    years = list(range(int(start[:4]), int(end[:4]) + 1))
    all_data = []
    
    for yr in years:
        s = f"{yr}0101"
        e = f"{yr}1231"
        if yr == int(start[:4]):
            s = start
        if yr == int(end[:4]):
            e = end
        
        print(f"  Year {yr}...")
        df = fetch_all(pro, "daily_basic", ts_code="", start_date=s, end_date=e,
                       fields="ts_code,trade_date,pe,pb,total_mv,circ_mv,turnover_rate,volume_ratio")
        if len(df) > 0:
            all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    result = pd.concat(all_data, ignore_index=True)
    return result


def download_index_daily(pro, start: str, end: str) -> pd.DataFrame:
    """Download daily index data for key benchmarks."""
    print(f"\n[index_daily] Downloading...")
    all_data = []
    
    for idx_code, idx_name in INDEX_CODES.items():
        print(f"  {idx_name} ({idx_code})...")
        df = fetch_all(pro, "index_daily", ts_code=idx_code, 
                       start_date=start, end_date=end,
                       fields="ts_code,trade_date,open,high,low,close,pre_close,vol,amount")
        if len(df) > 0:
            all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    result = pd.concat(all_data, ignore_index=True)
    return result


def download_index_classify(pro) -> pd.DataFrame:
    """Download industry classification (88-series / 三级行业)."""
    print(f"\n[index_classify] Downloading...")
    
    # Download all levels
    all_data = []
    for level in ["L1", "L2", "L3"]:
        df = fetch_all(pro, "index_classify", level=level,
                       src="SW2021")  # Shenwan 2021 classification
        if len(df) > 0:
            all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    result = pd.concat(all_data, ignore_index=True)
    print(f"  Downloaded {len(result):,} classification records")
    return result


def download_index_member(pro) -> pd.DataFrame:
    """Download index member data (for industry indices)."""
    print(f"\n[index_member] Downloading...")
    
    # Get SW industry indices
    classify = pro.index_classify(level="L3", src="SW2021")
    if classify is None or len(classify) == 0:
        print("  No industry classification found, skipping member download")
        return pd.DataFrame()
    
    index_codes = classify["index_code"].unique().tolist()
    print(f"  Found {len(index_codes)} industry indices")
    
    all_data = []
    for i, idx_code in enumerate(index_codes):
        if i % 20 == 0:
            print(f"  ... {i+1}/{len(index_codes)}")
        df = fetch_all(pro, "index_member", index_code=idx_code,
                       fields="index_code,index_name,con_code,con_name,in_date,out_date,is_new")
        if len(df) > 0:
            all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    result = pd.concat(all_data, ignore_index=True)
    print(f"  Downloaded {len(result):,} member records")
    return result


def download_hk_hold(pro, start: str, end: str) -> pd.DataFrame:
    """Download northbound holdings data."""
    print(f"\n[hk_hold] Downloading...")
    
    years = list(range(int(start[:4]), int(end[:4]) + 1))
    all_data = []
    
    for yr in years:
        s = f"{yr}0101"
        e = f"{yr}1231"
        if yr == int(start[:4]):
            s = start
        if yr == int(end[:4]):
            e = end
        
        print(f"  Year {yr}...")
        df = fetch_all(pro, "hk_hold", ts_code="", start_date=s, end_date=e,
                       fields="ts_code,trade_date,vol,hold_ratio")
        if len(df) > 0:
            all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    result = pd.concat(all_data, ignore_index=True)
    return result


def download_margin(pro, start: str, end: str) -> pd.DataFrame:
    """Download margin financing data."""
    print(f"\n[margin] Downloading...")
    
    years = list(range(int(start[:4]), int(end[:4]) + 1))
    all_data = []
    
    for yr in years:
        s = f"{yr}0101"
        e = f"{yr}1231"
        if yr == int(start[:4]):
            s = start
        if yr == int(end[:4]):
            e = end
        
        print(f"  Year {yr}...")
        df = fetch_all(pro, "margin", trade_date="", start_date=s, end_date=e,
                       fields="trade_date,exchange,name,rzye,rqye,rzmre,rqyl")
        if len(df) > 0:
            all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    result = pd.concat(all_data, ignore_index=True)
    return result


def download_fund_basic(pro) -> pd.DataFrame:
    """Download ETF basic info."""
    print(f"\n[fund_basic] Downloading ETFs...")
    df = pro.fund_basic(market="E")
    if df is None:
        return pd.DataFrame()
    print(f"  Downloaded {len(df):,} ETF records")
    return df


# ── DuckDB Catalog Build ────────────────────────────────────────────────────

def build_catalog():
    """Main entry point: download all data and build the DuckDB catalog."""
    
    pro = get_pro()
    today = datetime.now().strftime("%Y%m%d")
    
    print("=" * 60)
    print("A-SHARE DATA LAKE BUILDER")
    print(f"Start: {START_DATE} → End: {today}")
    print(f"Catalog: {CATALOG_PATH}")
    print("=" * 60)
    
    # Connect to DuckDB
    con = duckdb.connect(str(CATALOG_PATH))
    
    # Enable progress bar
    con.execute("SET enable_progress_bar = true")
    
    # ── Reference Data ──
    print("\n── Reference Data ──")
    
    # Stock basic
    stocks_basic = download_stock_basic(pro)
    if len(stocks_basic) > 0:
        con.execute("CREATE OR REPLACE TABLE stocks_basic AS SELECT * FROM stocks_basic")
        con.execute("CREATE INDEX IF NOT EXISTS idx_sb_ts_code ON stocks_basic(ts_code)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_sb_industry ON stocks_basic(industry)")
    
    # Trade calendar
    trade_cal = download_trade_cal(pro)
    if len(trade_cal) > 0:
        con.execute("CREATE OR REPLACE TABLE trade_cal AS SELECT * FROM trade_cal")
        con.execute("CREATE INDEX IF NOT EXISTS idx_tc_date ON trade_cal(cal_date)")
    
    # Index classification
    index_classify = download_index_classify(pro)
    if len(index_classify) > 0:
        con.execute("CREATE OR REPLACE TABLE index_classify AS SELECT * FROM index_classify")
    
    # ── Daily Market Data ──
    print("\n── Daily Market Data ──")
    
    # Stock daily OHLCV
    stocks_daily = download_daily(pro, START_DATE, today)
    if len(stocks_daily) > 0:
        con.execute("CREATE OR REPLACE TABLE stocks_daily AS SELECT * FROM stocks_daily")
        con.execute("CREATE INDEX IF NOT EXISTS idx_sd_ts ON stocks_daily(ts_code, trade_date)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_sd_date ON stocks_daily(trade_date)")
    
    # Adjustment factors
    adj_factor = download_adj_factor(pro, START_DATE, today)
    if len(adj_factor) > 0:
        con.execute("CREATE OR REPLACE TABLE adj_factor AS SELECT * FROM adj_factor")
        con.execute("CREATE INDEX IF NOT EXISTS idx_af_ts ON adj_factor(ts_code, trade_date)")
    
    # Daily basic (PE, PB, etc.)
    daily_basic = download_daily_basic(pro, START_DATE, today)
    if len(daily_basic) > 0:
        con.execute("CREATE OR REPLACE TABLE stocks_daily_basic AS SELECT * FROM daily_basic")
        con.execute("CREATE INDEX IF NOT EXISTS idx_sdb_ts ON stocks_daily_basic(ts_code, trade_date)")
    
    # ── Index Data ──
    print("\n── Index Data ──")
    
    # Index daily
    index_daily = download_index_daily(pro, START_DATE, today)
    if len(index_daily) > 0:
        con.execute("CREATE OR REPLACE TABLE index_daily AS SELECT * FROM index_daily")
        con.execute("CREATE INDEX IF NOT EXISTS idx_id_ts ON index_daily(ts_code, trade_date)")
    
    # Index members
    index_member = download_index_member(pro)
    if len(index_member) > 0:
        con.execute("CREATE OR REPLACE TABLE index_member AS SELECT * FROM index_member")
    
    # ── Flow Data ──
    print("\n── Flow Data ──")
    
    # Northbound holdings
    hk_hold = download_hk_hold(pro, START_DATE, today)
    if len(hk_hold) > 0:
        con.execute("CREATE OR REPLACE TABLE hk_hold AS SELECT * FROM hk_hold")
        con.execute("CREATE INDEX IF NOT EXISTS idx_hk_ts ON hk_hold(ts_code, trade_date)")
    
    # Margin financing
    margin = download_margin(pro, START_DATE, today)
    if len(margin) > 0:
        con.execute("CREATE OR REPLACE TABLE margin AS SELECT * FROM margin")
    
    # ── Fund/ETF Data ──
    print("\n── Fund/ETF Data ──")
    
    # Fund basic
    fund_basic = download_fund_basic(pro)
    if len(fund_basic) > 0:
        con.execute("CREATE OR REPLACE TABLE fund_basic AS SELECT * FROM fund_basic")
    
    # ── Build Adjusted Price View ──
    print("\n── Building Adjusted Prices ──")
    con.execute("""
        CREATE OR REPLACE VIEW stocks_daily_adj AS
        SELECT 
            d.ts_code,
            d.trade_date,
            d.open * COALESCE(a.adj_factor, 1.0) AS open_adj,
            d.high * COALESCE(a.adj_factor, 1.0) AS high_adj,
            d.low * COALESCE(a.adj_factor, 1.0) AS low_adj,
            d.close * COALESCE(a.adj_factor, 1.0) AS close_adj,
            d.pre_close * COALESCE(a.adj_factor, 1.0) AS pre_close_adj,
            d.vol,
            d.amount
        FROM stocks_daily d
        LEFT JOIN adj_factor a 
            ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
    """)
    
    # ── Build Master Table View ──
    print("── Building Master View ──")
    con.execute("""
        CREATE OR REPLACE VIEW master_view AS
        SELECT 
            d.ts_code,
            d.trade_date,
            d.open_adj,
            d.high_adj,
            d.low_adj,
            d.close_adj,
            d.pre_close_adj,
            d.vol,
            d.amount,
            b.pe,
            b.pb,
            b.total_mv,
            b.circ_mv,
            b.turnover_rate,
            s.name,
            s.industry,
            s.list_date,
            s.delist_date,
            -- ST flag: name contains ST or *ST
            CASE WHEN s.name LIKE '%ST%' THEN 1 ELSE 0 END AS is_st,
            -- Listed days
            DATEDIFF('day', s.list_date::DATE, d.trade_date::DATE) AS listed_days
        FROM stocks_daily_adj d
        LEFT JOIN stocks_daily_basic b
            ON d.ts_code = b.ts_code AND d.trade_date = b.trade_date
        LEFT JOIN stocks_basic s
            ON d.ts_code = s.ts_code
        WHERE d.close_adj IS NOT NULL AND d.close_adj > 0
    """)
    
    # ── Summary ──
    print("\n── Catalog Summary ──")
    tables = con.execute("""
        SELECT table_name, 
               estimated_size,
               CASE WHEN table_type = 'VIEW' THEN 'VIEW' ELSE 'TABLE' END as tbl_type
        FROM duckdb_tables()
        UNION ALL
        SELECT view_name, NULL, 'VIEW'
        FROM duckdb_views()
        ORDER BY table_name
    """).fetchall()
    
    for name, size, tbl_type in tables:
        if tbl_type == 'TABLE':
            row_count = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            print(f"  {name:25s} ({tbl_type:5s}): {row_count:>10,} rows")
        else:
            print(f"  {name:25s} ({tbl_type:5s})")
    
    # Write manifest
    manifest = {
        "built_at": datetime.now().isoformat(),
        "start_date": START_DATE,
        "end_date": today,
        "tables": [t[0] for t in tables if t[2] == 'TABLE'],
        "views": [t[0] for t in tables if t[2] == 'VIEW'],
    }
    
    manifest_path = MANIFEST_DIR / f"build_{today}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written: {manifest_path}")
    
    con.close()
    print("\n✅ Data lake build complete.")


if __name__ == "__main__":
    build_catalog()
