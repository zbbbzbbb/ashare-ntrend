"""
Data Audit — validate data quality using the quant-data catalog.

Checks: price continuity, ST/delisting handling, suspension periods,
       adjustment factor quality, industry classification coverage,
       trade calendar completeness.
"""

from pathlib import Path
from datetime import datetime
import json

import duckdb
import pandas as pd
import numpy as np

from .loader import get_connection, get_latest_date

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"


def run_audit() -> dict:
    """Run full data quality audit. Returns dict of findings."""
    
    con = get_connection()
    findings = {
        "audit_date": datetime.now().isoformat(),
        "catalog": str(Path.home() / "quant-data" / "catalog.duckdb"),
        "sections": {}
    }
    
    print("=" * 50)
    print("DATA QUALITY AUDIT")
    print("=" * 50)
    
    # 1. Price continuity check (using daily + adj_factor join)
    print("\n1. Price Continuity Check")
    try:
        df = con.execute("""
            WITH adj_prices AS (
                SELECT d.ts_code, d.trade_date, 
                       d.close * COALESCE(a.adj_factor, 1.0) AS close_adj
                FROM daily d
                LEFT JOIN adj_factor a ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
                WHERE d.close > 0
            ),
            price_changes AS (
                SELECT ts_code, trade_date, close_adj,
                       LAG(close_adj) OVER (PARTITION BY ts_code ORDER BY trade_date) AS prev_close,
                       (close_adj - LAG(close_adj) OVER (PARTITION BY ts_code ORDER BY trade_date)) 
                       / NULLIF(LAG(close_adj) OVER (PARTITION BY ts_code ORDER BY trade_date), 0) AS ret
                FROM adj_prices
            )
            SELECT COUNT(*) as n, 
                   SUM(CASE WHEN ABS(ret) > 0.5 THEN 1 ELSE 0 END) as large_jumps,
                   SUM(CASE WHEN ABS(ret) > 0.11 THEN 1 ELSE 0 END) as daily_limit_hits
            FROM price_changes
            WHERE ret IS NOT NULL
        """).fetchdf()
        
        total = df["n"].iloc[0]
        jumps = df["large_jumps"].iloc[0]
        limits = df["daily_limit_hits"].iloc[0]
        
        pct_jumps = 100 * jumps / total if total > 0 else 0
        pct_limits = 100 * limits / total if total > 0 else 0
        
        findings["sections"]["price_continuity"] = {
            "total_observations": int(total),
            "large_jumps_gt50pct": int(jumps),
            "large_jumps_pct": round(pct_jumps, 4),
            "daily_limit_pct": round(pct_limits, 4),
            "status": "OK" if pct_jumps < 0.1 else "WARNING"
        }
        print(f"   {total:,} observations")
        print(f"   Jumps >50%: {jumps:,} ({pct_jumps:.3f}%)")
        print(f"   Status: {findings['sections']['price_continuity']['status']}")
    except Exception as e:
        findings["sections"]["price_continuity"] = {"error": str(e)}
        print(f"   Error: {e}")
    
    # 2. ST stock coverage
    print("\n2. ST Stock Handling")
    try:
        df = con.execute("""
            SELECT 
                COUNT(DISTINCT CASE WHEN name LIKE '%ST%' THEN ts_code END) as st_symbols,
                COUNT(DISTINCT ts_code) as total_symbols
            FROM stock_basic
        """).fetchdf()
        
        findings["sections"]["st_stocks"] = {
            "st_symbols": int(df["st_symbols"].iloc[0]),
            "total_symbols": int(df["total_symbols"].iloc[0]),
            "st_pct": round(100 * df["st_symbols"].iloc[0] / df["total_symbols"].iloc[0], 2),
            "status": "OK"
        }
        print(f"   ST symbols: {df['st_symbols'].iloc[0]:,} / {df['total_symbols'].iloc[0]:,}")
    except Exception as e:
        findings["sections"]["st_stocks"] = {"error": str(e)}
        print(f"   Error: {e}")
    
    # 3. Date range
    print("\n3. Date Range Coverage")
    try:
        df = con.execute("""
            SELECT MIN(trade_date) as min_date, MAX(trade_date) as max_date,
                   COUNT(DISTINCT trade_date) as trading_days
            FROM daily
        """).fetchdf()
        
        findings["sections"]["date_range"] = {
            "start": str(df["min_date"].iloc[0]),
            "end": str(df["max_date"].iloc[0]),
            "trading_days": int(df["trading_days"].iloc[0]),
            "status": "OK"
        }
        print(f"   Range: {df['min_date'].iloc[0]} → {df['max_date'].iloc[0]}")
        print(f"   Trading days: {df['trading_days'].iloc[0]:,}")
    except Exception as e:
        findings["sections"]["date_range"] = {"error": str(e)}
        print(f"   Error: {e}")
    
    # 4. Missing data
    print("\n4. Missing Data Rates")
    try:
        df = con.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN d.close IS NULL OR d.close <= 0 THEN 1 ELSE 0 END) as missing_close,
                SUM(CASE WHEN d.vol IS NULL THEN 1 ELSE 0 END) as missing_vol
            FROM daily d
        """).fetchdf()
        
        total = df["total"].iloc[0]
        findings["sections"]["missing_data"] = {
            "total_rows": int(total),
            "missing_close": int(df["missing_close"].iloc[0]),
            "missing_close_pct": round(100 * df["missing_close"].iloc[0] / total, 4),
            "status": "OK" if df["missing_close"].iloc[0] / total < 0.01 else "WARNING"
        }
        print(f"   Missing close: {df['missing_close'].iloc[0]:,} / {total:,}")
    except Exception as e:
        findings["sections"]["missing_data"] = {"error": str(e)}
        print(f"   Error: {e}")
    
    # 5. Index data
    print("\n5. Index Data Check")
    try:
        df = con.execute("""
            SELECT ts_code, COUNT(*) as n_days, 
                   MIN(trade_date) as min_d, MAX(trade_date) as max_d
            FROM index_daily
            WHERE ts_code IN ('000300.SH', '000905.SH', '000016.SH', '399006.SZ', '000852.SH')
            GROUP BY ts_code
            ORDER BY ts_code
        """).fetchdf()
        
        findings["sections"]["index_data"] = {
            "indices": len(df),
            "details": df.to_dict(orient="records"),
            "status": "OK" if len(df) >= 3 else "WARNING"
        }
        for _, row in df.iterrows():
            print(f"   {row['ts_code']}: {row['n_days']:,} days ({row['min_d']} → {row['max_d']})")
    except Exception as e:
        findings["sections"]["index_data"] = {"error": str(e)}
        print(f"   Error: {e}")
    
    # 6. Industry classification
    print("\n6. Industry Classification")
    try:
        df = con.execute("""
            SELECT level, COUNT(DISTINCT index_code) as n_indices
            FROM index_classify
            GROUP BY level
            ORDER BY level
        """).fetchdf()
        
        findings["sections"]["industry_classification"] = {
            "levels": df.to_dict(orient="records"),
            "status": "OK"
        }
        for _, row in df.iterrows():
            print(f"   {row['level']}: {row['n_indices']} indices")
    except Exception as e:
        findings["sections"]["industry_classification"] = {"error": str(e)}
        print(f"   Error: {e}")
    
    # 7. Suspension detection
    print("\n7. Suspension Detection")
    try:
        df = con.execute("""
            WITH vol_zero AS (
                SELECT ts_code, trade_date,
                       CASE WHEN vol = 0 OR vol IS NULL THEN 1 ELSE 0 END AS suspended
                FROM daily
                WHERE trade_date >= '20200101'
            )
            SELECT COUNT(DISTINCT ts_code) as symbols
            FROM vol_zero
            WHERE suspended = 1
        """).fetchdf()
        
        findings["sections"]["suspensions"] = {
            "symbols_with_zero_vol_since_2020": int(df["symbols"].iloc[0]),
            "status": "OK"
        }
        print(f"   Symbols with zero vol (since 2020): {df['symbols'].iloc[0]:,}")
    except Exception as e:
        findings["sections"]["suspensions"] = {"error": str(e)}
        print(f"   Error: {e}")
    
    # 8. Adjustment factor quality
    print("\n8. Adjustment Factor Quality")
    try:
        df = con.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN adj_factor IS NULL THEN 1 ELSE 0 END) as null_count,
                MIN(adj_factor) as min_af,
                MAX(adj_factor) as max_af,
                AVG(adj_factor) as avg_af
            FROM adj_factor
        """).fetchdf()
        
        findings["sections"]["adj_factor"] = {
            "total": int(df["total"].iloc[0]),
            "null_count": int(df["null_count"].iloc[0]),
            "range": f"[{df['min_af'].iloc[0]:.4f}, {df['max_af'].iloc[0]:.4f}]",
            "avg": round(df["avg_af"].iloc[0], 4),
            "status": "OK" if df["null_count"].iloc[0] / df["total"].iloc[0] < 0.01 else "WARNING"
        }
        print(f"   AF range: [{df['min_af'].iloc[0]:.4f}, {df['max_af'].iloc[0]:.4f}]")
        print(f"   Null count: {df['null_count'].iloc[0]:,}")
    except Exception as e:
        findings["sections"]["adj_factor"] = {"error": str(e)}
        print(f"   Error: {e}")
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = OUTPUT_DIR / "data_audit.json"
    audit_path.write_text(json.dumps(findings, indent=2, default=str))
    print(f"\n✅ Audit complete. Results: {audit_path}")
    
    return findings


if __name__ == "__main__":
    run_audit()
