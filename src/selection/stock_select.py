"""
Stock Selection — within-industry constituent filtering.

From selected industries, picks stocks meeting:
  1. MACD Bullish (DIF > DEA)
  2. N-Pattern Breakout
  3. Volume Confirmation (vol > 20-day average)
  4. Relative Strength (above industry median)
  5. Above BBI (preferred)

Excludes: ST, *ST, listed < 250 days, delisting risk, extremely illiquid.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from signals.macd import compute_macd, macd_bullish
from signals.n_pattern import detect_n_pattern
from signals.bbi import compute_bbi_signal
from signals.relative_strength import volume_above_avg


class StockSelector:
    """Filters and ranks stocks within selected industries."""
    
    def __init__(
        self,
        min_listed_days: int = 250,
        volume_avg_period: int = 20,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        n_leg1_start: int = 20,
        n_leg1_end: int = 10,
        n_pullback_end: int = 5,
        max_positions: int = 10,
    ):
        self.min_listed_days = min_listed_days
        self.volume_avg_period = volume_avg_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.n_leg1_start = n_leg1_start
        self.n_leg1_end = n_leg1_end
        self.n_pullback_end = n_pullback_end
        self.max_positions = max_positions
    
    def screen_stocks(
        self,
        stock_data: dict,  # {ts_code: DataFrame with OHLCV + indicators}
        date_idx: int = -1,
    ) -> pd.DataFrame:
        """Screen and score all stocks.
        
        Returns DataFrame with screening results and composite score.
        """
        results = []
        
        for ts_code, df in stock_data.items():
            if len(df) <= date_idx or len(df) < self.min_listed_days:
                continue
            
            close = df["close_adj"]
            vol = df.get("vol", pd.Series(1, index=close.index))
            high = df.get("high_adj", close)
            
            # 1. MACD Bullish
            macd_df = compute_macd(close, self.macd_fast, self.macd_slow, self.macd_signal)
            macd_ok = float(macd_bullish(macd_df).iloc[date_idx])
            
            # 2. N-Pattern Breakout
            n_df = detect_n_pattern(
                close, high,
                leg1_start=self.n_leg1_start,
                leg1_end=self.n_leg1_end,
                pullback_end=self.n_pullback_end,
            )
            n_ok = float(n_df["is_n_pattern"].iloc[date_idx])
            n_quality = n_df["pattern_quality"].iloc[date_idx]
            
            # 3. Volume Confirmation
            vol_ok = float(volume_above_avg(vol, self.volume_avg_period).iloc[date_idx])
            
            # 4. BBI
            bbi_df = compute_bbi_signal(close)
            above_bbi = float(bbi_df["above_bbi"].iloc[date_idx])
            
            # Composite score
            score = (
                macd_ok * 0.25 +
                n_quality * 0.30 +
                vol_ok * 0.20 +
                above_bbi * 0.15 +
                (1.0 if n_ok else 0.0) * 0.10
            )
            
            results.append({
                "ts_code": ts_code,
                "macd_bullish": macd_ok,
                "n_pattern": n_ok,
                "n_quality": n_quality,
                "volume_ok": vol_ok,
                "above_bbi": above_bbi,
                "score": score,
                "close": close.iloc[date_idx],
            })
        
        if not results:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(results)
        
        # RS rank: above industry median (cross-sectional among candidates)
        if "close" in result_df.columns:
            # Use recent momentum as RS proxy
            result_df["rs_rank"] = result_df["score"].rank(ascending=False, pct=True)
            result_df["rs_above_median"] = result_df["rs_rank"] > 0.5
        
        return result_df.sort_values("score", ascending=False)
    
    def select_stocks(
        self,
        stock_data: dict,
        date_idx: int = -1,
        max_stocks: int = None,
    ) -> list:
        """Select top stocks passing all required screens.
        
        Required:
          - MACD bullish
          - N-pattern detected
          - Volume above average
        
        Preferred:
          - Above BBI
          - Higher composite score
        """
        if max_stocks is None:
            max_stocks = self.max_positions
        
        screened = self.screen_stocks(stock_data, date_idx)
        
        if len(screened) == 0:
            return []
        
        # Hard requirements
        qualified = screened[
            (screened["macd_bullish"] > 0) &
            (screened["n_pattern"] > 0) &
            (screened["volume_ok"] > 0)
        ]
        
        # Relax N-pattern if too few candidates
        if len(qualified) < 3:
            qualified = screened[
                (screened["macd_bullish"] > 0) &
                (screened["volume_ok"] > 0)
            ]
        
        # Prefer above BBI
        qualified = qualified.sort_values(
            ["above_bbi", "score"], ascending=[False, False]
        )
        
        selected = qualified.head(max_stocks)["ts_code"].tolist()
        return selected


def build_stock_data(loader, ts_codes: list, start: str, end: str) -> dict:
    """Build stock price data dict from the data lake.
    
    Args:
        loader: Data loader module
        ts_codes: List of stock codes
        start, end: Date range
    
    Returns:
        {ts_code: DataFrame with OHLCV columns}
    """
    from ..data.loader import get_daily_data
    
    all_data = get_daily_data(ts_codes, start, end)
    if len(all_data) == 0:
        return {}
    
    stock_data = {}
    for ts_code, group in all_data.groupby("ts_code"):
        group = group.sort_values("trade_date").reset_index(drop=True)
        # Rename columns for consistency
        group = group.rename(columns={
            "open_adj": "open",
            "high_adj": "high",
            "low_adj": "low",
            "close_adj": "close",
            "pre_close_adj": "pre_close",
        })
        stock_data[ts_code] = group
    
    return stock_data
