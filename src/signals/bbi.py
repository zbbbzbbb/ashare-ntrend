"""
BBI (Bull and Bear Index) — multi-period moving average composite.

BBI = (MA3 + MA6 + MA12 + MA24) / 4

Used as a trend filter:
  - Price > BBI → bullish trend environment (preferred for entries)
  - Price < BBI → bearish trend environment

Also serves as a dynamic stop-loss level:
  - Entry above BBI: exit when two consecutive daily closes < BBI
  - Entry below BBI: exit when price falls 3% below entry cost
"""

import numpy as np
import pandas as pd


def compute_bbi(close: pd.Series, periods: tuple = (3, 6, 12, 24)) -> pd.Series:
    """Compute BBI for a price series.
    
    Args:
        close: Series of adjusted closing prices
        periods: MA periods (default 3, 6, 12, 24)
    
    Returns:
        Series of BBI values
    """
    mas = [close.rolling(window=p).mean() for p in periods]
    bbi = sum(mas) / len(mas)
    return bbi


def compute_bbi_signal(close: pd.Series) -> pd.DataFrame:
    """Compute BBI and derived signals.
    
    Returns DataFrame with:
      - bbi: BBI value
      - above_bbi: Price > BBI
      - below_bbi: Price < BBI
      - bbi_distance_pct: Percentage distance from BBI
      - two_day_below: Two consecutive closes below BBI (exit signal)
    """
    bbi = compute_bbi(close)
    above_bbi = close > bbi
    below_bbi = close < bbi
    bbi_distance = (close / bbi) - 1.0
    
    # Two consecutive closes below BBI
    two_day_below = below_bbi & below_bbi.shift(1)
    
    return pd.DataFrame({
        "bbi": bbi,
        "above_bbi": above_bbi,
        "below_bbi": below_bbi,
        "bbi_distance_pct": bbi_distance * 100,
        "two_day_below_bbi": two_day_below,
    }, index=close.index)


class BBIStop:
    """BBI-based stop-loss manager for individual positions.
    
    Case A (Entry below BBI): Exit when price falls -3% below entry cost.
    Case B (Entry above BBI): Exit when two consecutive closes < BBI.
    """
    
    def __init__(self):
        self.positions = {}  # ts_code → {entry_price, entry_date, entry_above_bbi, ...}
    
    def add_position(self, ts_code: str, entry_price: float, entry_date, 
                     entry_above_bbi: bool, bbi_value: float):
        """Record a new position entry."""
        self.positions[ts_code] = {
            "entry_price": entry_price,
            "entry_date": entry_date,
            "entry_above_bbi": entry_above_bbi,
            "bbi_value_at_entry": bbi_value,
        }
    
    def check_exit(self, ts_code: str, current_price: float, 
                   bbi_df: pd.DataFrame, idx: int) -> tuple[bool, str]:
        """Check if position should be exited.
        
        Returns:
            (should_exit: bool, reason: str)
        """
        if ts_code not in self.positions:
            return False, "no_position"
        
        pos = self.positions[ts_code]
        pnl_pct = (current_price / pos["entry_price"] - 1.0) * 100
        
        if pos["entry_above_bbi"]:
            # Case B: Two consecutive closes below BBI
            two_day = bbi_df["two_day_below_bbi"].iloc[idx] if idx < len(bbi_df) else False
            if two_day:
                return True, f"bbi_exit_two_day_below (PnL: {pnl_pct:.1f}%)"
        else:
            # Case A: -3% from entry
            if pnl_pct <= -3.0:
                return True, f"stop_loss_3pct (PnL: {pnl_pct:.1f}%)"
        
        return False, "hold"
    
    def remove_position(self, ts_code: str):
        """Remove a closed position."""
        self.positions.pop(ts_code, None)
