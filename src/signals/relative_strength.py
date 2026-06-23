"""
Relative Strength & Volume indicators for stock/industry ranking.
"""

import numpy as np
import pandas as pd


def relative_strength(close: pd.Series, benchmark: pd.Series = None, 
                      period: int = 20) -> pd.Series:
    """Compute relative strength vs benchmark over a lookback period.
    
    RS = (1 + stock_return) / (1 + benchmark_return) - 1
    
    If no benchmark provided, computes absolute momentum (price / price[N] - 1).
    """
    if benchmark is None:
        return close / close.shift(period) - 1.0
    
    stock_ret = close / close.shift(period) - 1.0
    bench_ret = benchmark / benchmark.shift(period) - 1.0
    return (1 + stock_ret) / (1 + bench_ret) - 1.0


def rs_rank(rs_series: pd.Series) -> pd.Series:
    """Convert relative strength to percentile rank (0-100, higher = stronger)."""
    return rs_series.rank(pct=True) * 100


def volume_expansion(vol: pd.Series, avg_period: int = 60) -> pd.Series:
    """Check if current volume exceeds the N-day average.
    
    Returns: ratio of current vol to avg vol (>1 = expanding).
    """
    avg_vol = vol.rolling(window=avg_period).mean()
    return vol / avg_vol


def volume_above_avg(vol: pd.Series, avg_period: int = 20) -> pd.Series:
    """Boolean: volume > N-day average."""
    return vol > vol.rolling(window=avg_period).mean()


def turnover_expansion(turnover_rate: pd.Series, avg_period: int = 60) -> pd.Series:
    """Check if turnover rate exceeds N-day average."""
    avg_turnover = turnover_rate.rolling(window=avg_period).mean()
    return turnover_rate / avg_turnover


def capital_inflow_proxy(vol: pd.Series, close: pd.Series, period: int = 10) -> pd.Series:
    """Simple capital inflow proxy: volume expansion on up days.
    
    Compares average volume on up days vs down days over the period.
    Values > 1 suggest net buying pressure.
    """
    up_days = close > close.shift(1)
    
    up_vol = vol.where(up_days, 0).rolling(window=period).mean()
    down_vol = vol.where(~up_days, 0).rolling(window=period).mean()
    
    return up_vol / down_vol.replace(0, np.nan)


def rs_momentum_composite(close: pd.Series, benchmark: pd.Series = None,
                          periods: tuple = (5, 10, 20, 60)) -> pd.DataFrame:
    """Multi-period relative strength composite.
    
    Returns DataFrame with RS for each period and a composite average.
    """
    results = {}
    for p in periods:
        results[f"rs_{p}d"] = relative_strength(close, benchmark, p)
    
    df = pd.DataFrame(results, index=close.index)
    df["rs_composite"] = df.mean(axis=1)
    
    return df
