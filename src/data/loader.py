"""
Data Loader — unified query interface for the quant-data DuckDB catalog.

Catalog at ~/quant-data/catalog.duckdb.

All date inputs/outputs use YYYYMMDD string format. 
Internally converts to YYYY-MM-DD for DuckDB DATE type queries.
"""

from pathlib import Path
from typing import Optional, List, Tuple

import duckdb
import pandas as pd

CATALOG_PATH = Path.home() / "quant-data" / "catalog.duckdb"

_conn: Optional[duckdb.DuckDBPyConnection] = None


def get_connection(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Get or create a cached DuckDB connection."""
    global _conn
    if _conn is None:
        _conn = duckdb.connect(str(CATALOG_PATH), read_only=read_only)
    return _conn


def close():
    """Close the cached connection."""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


# ── Date Helpers ────────────────────────────────────────────────────────────

def _iso(d: str) -> str:
    """YYYYMMDD → YYYY-MM-DD."""
    if "-" in d:
        return d
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def _ymd(d) -> str:
    """Any date → YYYYMMDD string."""
    s = str(d)[:10]
    return s.replace("-", "")


# ── Calendar ───────────────────────────────────────────────────────────────

def get_trade_dates(start: str = "20050101", end: str = "20991231") -> pd.DataFrame:
    """Get open trading dates in range. Returns dates as YYYYMMDD strings."""
    con = get_connection()
    df = con.execute("""
        SELECT cal_date FROM trade_cal 
        WHERE is_open = 1 AND cal_date >= ? AND cal_date <= ?
        ORDER BY cal_date
    """, [_iso(start), _iso(end)]).fetchdf()
    if len(df) > 0:
        df["cal_date"] = df["cal_date"].apply(_ymd)
    return df


# ── Stock Daily (Adjusted) ─────────────────────────────────────────────────

def get_daily_data(ts_codes: List[str], start: str, end: str) -> pd.DataFrame:
    """Get adjusted daily OHLCV for a list of stocks.
    
    Prices adjusted by raw price × adj_factor.
    Returns trade_date as YYYYMMDD strings.
    """
    con = get_connection()
    if not ts_codes:
        return pd.DataFrame()
    
    placeholders = ",".join(["?"] * len(ts_codes))
    query = f"""
        SELECT d.ts_code, d.trade_date,
               d.open * COALESCE(a.adj_factor, 1.0) AS open_adj,
               d.high * COALESCE(a.adj_factor, 1.0) AS high_adj,
               d.low * COALESCE(a.adj_factor, 1.0) AS low_adj,
               d.close * COALESCE(a.adj_factor, 1.0) AS close_adj,
               d.pre_close * COALESCE(a.adj_factor, 1.0) AS pre_close_adj,
               d.vol, d.amount
        FROM daily d
        LEFT JOIN adj_factor a ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
        WHERE d.ts_code IN ({placeholders})
          AND d.trade_date >= ? AND d.trade_date <= ?
        ORDER BY d.ts_code, d.trade_date
    """
    df = con.execute(query, ts_codes + [_iso(start), _iso(end)]).fetchdf()
    if len(df) > 0:
        df["trade_date"] = df["trade_date"].apply(_ymd)
    return df


def get_daily_data_with_basic(ts_codes: List[str], start: str, end: str) -> pd.DataFrame:
    """Get adjusted daily data joined with daily_basic."""
    con = get_connection()
    if not ts_codes:
        return pd.DataFrame()
    
    placeholders = ",".join(["?"] * len(ts_codes))
    query = f"""
        SELECT d.ts_code, d.trade_date,
               d.open * COALESCE(a.adj_factor, 1.0) AS open_adj,
               d.high * COALESCE(a.adj_factor, 1.0) AS high_adj,
               d.low * COALESCE(a.adj_factor, 1.0) AS low_adj,
               d.close * COALESCE(a.adj_factor, 1.0) AS close_adj,
               d.pre_close * COALESCE(a.adj_factor, 1.0) AS pre_close_adj,
               d.vol, d.amount,
               b.pe, b.pb, b.total_mv, b.circ_mv,
               b.turnover_rate, b.volume_ratio, b.total_share, b.float_share
        FROM daily d
        LEFT JOIN adj_factor a ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
        LEFT JOIN daily_basic b ON d.ts_code = b.ts_code AND d.trade_date = b.trade_date
        WHERE d.ts_code IN ({placeholders})
          AND d.trade_date >= ? AND d.trade_date <= ?
        ORDER BY d.ts_code, d.trade_date
    """
    df = con.execute(query, ts_codes + [_iso(start), _iso(end)]).fetchdf()
    if len(df) > 0:
        df["trade_date"] = df["trade_date"].apply(_ymd)
    return df


# ── Stock Universe ─────────────────────────────────────────────────────────

def get_stock_universe(date_str: str, exclude_st: bool = True,
                       min_listed_days: int = 250,
                       min_market_cap: float = 200000) -> pd.DataFrame:
    """Get investable stock universe on a given date (YYYYMMDD).
    
    Args:
        date_str: Trade date in YYYYMMDD format
        exclude_st: Exclude ST/*ST stocks
        min_listed_days: Minimum days since listing
        min_market_cap: Minimum circ_mv in 万元 (default 200000 = 2B yuan)
    """
    con = get_connection()
    d = _iso(date_str)
    
    query = """
        SELECT d.ts_code, s.name, s.industry, s.list_date, s.delist_date, s.list_status,
               d.close * COALESCE(a.adj_factor, 1.0) AS close_adj,
               d.vol, d.amount, b.total_mv, b.circ_mv, b.turnover_rate, b.pe, b.pb,
               DATEDIFF('day', s.list_date::DATE, d.trade_date::DATE) AS listed_days,
               CASE WHEN s.name LIKE '%ST%' THEN 1 ELSE 0 END AS is_st
        FROM daily d
        LEFT JOIN adj_factor a ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
        LEFT JOIN stock_basic s ON d.ts_code = s.ts_code
        LEFT JOIN daily_basic b ON d.ts_code = b.ts_code AND d.trade_date = b.trade_date
        WHERE d.trade_date = ?
          AND d.close > 0 AND s.list_date IS NOT NULL
          AND DATEDIFF('day', s.list_date::DATE, d.trade_date::DATE) >= ?
          AND COALESCE(b.circ_mv, 0) >= ?
    """
    params = [d, min_listed_days, min_market_cap]
    
    if exclude_st:
        query += " AND s.name NOT LIKE '%ST%'"
    
    query += " ORDER BY COALESCE(b.circ_mv, 0) DESC"
    
    return con.execute(query, params).fetchdf()


def get_stock_basic() -> pd.DataFrame:
    """Get all stock basic info."""
    return get_connection().execute("SELECT * FROM stock_basic").fetchdf()


# ── Index Data ─────────────────────────────────────────────────────────────

def get_index_daily(index_code: str, start: str, end: str) -> pd.DataFrame:
    """Get daily OHLCV for an index. Returns trade_date as YYYYMMDD."""
    con = get_connection()
    df = con.execute("""
        SELECT * FROM index_daily
        WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
    """, [index_code, _iso(start), _iso(end)]).fetchdf()
    if len(df) > 0:
        df["trade_date"] = df["trade_date"].apply(_ymd)
    return df


def get_industry_indices(level: str = "L3", src: str = "SW2021") -> pd.DataFrame:
    """Get industry index codes at a given classification level."""
    con = get_connection()
    return con.execute("""
        SELECT * FROM index_classify
        WHERE level = ? AND src = ?
        ORDER BY index_code
    """, [level, src]).fetchdf()


def get_index_members(l3_code: str, date_str: str) -> pd.DataFrame:
    """Get constituents of an L3 industry index on a given date."""
    con = get_connection()
    return con.execute("""
        SELECT * FROM index_member_all
        WHERE l3_code = ? AND in_date <= ? AND (out_date IS NULL OR out_date > ?)
        ORDER BY ts_code
    """, [l3_code, _iso(date_str), _iso(date_str)]).fetchdf()


def get_industry_stocks(l3_codes: List[str], date_str: str) -> List[str]:
    """Get all constituent stock codes for L3 industries."""
    if not l3_codes:
        return []
    con = get_connection()
    placeholders = ",".join(["?"] * len(l3_codes))
    d = _iso(date_str)
    df = con.execute(f"""
        SELECT DISTINCT ts_code FROM index_member_all
        WHERE l3_code IN ({placeholders})
          AND in_date <= ? AND (out_date IS NULL OR out_date > ?)
    """, l3_codes + [d, d]).fetchdf()
    return df["ts_code"].tolist() if len(df) > 0 else []


def get_all_l3_index_codes() -> List[str]:
    """Get list of all L3 industry index codes (SW2021)."""
    df = get_industry_indices("L3", "SW2021")
    if len(df) == 0:
        con = get_connection()
        df = con.execute(
            "SELECT DISTINCT index_code FROM index_classify WHERE level = 'L3'"
        ).fetchdf()
    return df["index_code"].tolist() if len(df) > 0 else []


# ── Index Membership (Industry → Stocks) ───────────────────────────────────

def get_l3_to_stocks_map(date_str: str, l3_codes: List[str] = None) -> dict:
    """Build dict mapping l3_code → list of ts_code constituents."""
    con = get_connection()
    d = _iso(date_str)
    
    if l3_codes:
        placeholders = ",".join(["?"] * len(l3_codes))
        df = con.execute(f"""
            SELECT l3_code, ts_code FROM index_member_all
            WHERE l3_code IN ({placeholders})
              AND in_date <= ? AND (out_date IS NULL OR out_date > ?)
        """, l3_codes + [d, d]).fetchdf()
    else:
        df = con.execute("""
            SELECT l3_code, ts_code FROM index_member_all
            WHERE in_date <= ? AND (out_date IS NULL OR out_date > ?)
        """, [d, d]).fetchdf()
    
    result = {}
    for l3, group in df.groupby("l3_code"):
        result[l3] = group["ts_code"].tolist()
    return result


# ── Utility ────────────────────────────────────────────────────────────────

def get_latest_date(view: str = "daily") -> str:
    """Get the most recent trade date as YYYYMMDD."""
    con = get_connection()
    result = con.execute(f"SELECT MAX(trade_date) FROM {view}").fetchone()
    return _ymd(result[0]) if result and result[0] else "20050101"


def get_date_range(view: str = "daily") -> tuple:
    """Get (min_date, max_date) as YYYYMMDD strings."""
    con = get_connection()
    result = con.execute(
        f"SELECT MIN(trade_date), MAX(trade_date) FROM {view}"
    ).fetchone()
    return (_ymd(result[0]), _ymd(result[1])) if result else ("20050101", "20250101")


def table_exists(name: str) -> bool:
    """Check if a table/view exists in the catalog."""
    con = get_connection()
    try:
        con.execute(f"SELECT 1 FROM {name} LIMIT 0")
        return True
    except Exception:
        return False
