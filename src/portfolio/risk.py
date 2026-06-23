"""
Risk Management — BBI-based stop-loss and profit-taking rules.

Two-tier exit system:
  Case A (Entry below BBI): Exit at -3% from entry cost.
  Case B (Entry above BBI): Exit when two consecutive closes < BBI.

Profit Taking:
  Default: 20% gain → sell 50%, hold remainder to trend exit.
  Variant A: 20% gain → reduce 50%.
  Variant B: 20% → reduce 30%, 35% → reduce additional 30%, remainder trend.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from signals.bbi import compute_bbi_signal


class RiskManager:
    """Manages stop-loss and profit-taking for all open positions."""
    
    def __init__(self, profit_taking_variant: str = "default"):
        """
        Args:
            profit_taking_variant: 'default', 'variant_a', or 'variant_b'
        """
        self.variant = profit_taking_variant
        self.positions = {}  # ts_code → position dict
        
        # Profit-taking thresholds
        if profit_taking_variant == "variant_b":
            self.profit_rules = [
                (0.20, 0.30),  # 20% gain → sell 30%
                (0.35, 0.30),  # 35% gain → sell additional 30%
            ]
        else:
            self.profit_rules = [
                (0.20, 0.50),  # 20% gain → sell 50%
            ]
    
    def open_position(self, ts_code: str, entry_price: float, entry_date,
                      shares: int, above_bbi: bool, bbi_value: float):
        """Record a new position."""
        self.positions[ts_code] = {
            "entry_price": entry_price,
            "entry_date": entry_date,
            "shares": shares,
            "entry_above_bbi": above_bbi,
            "bbi_at_entry": bbi_value,
            "profit_taken_levels": set(),  # Track which levels have been taken
            "highest_price": entry_price,
        }
    
    def update(self, ts_code: str, current_price: float, current_bbi_df: pd.DataFrame,
               idx: int) -> dict:
        """Update position and check for exit signals.
        
        Args:
            idx: Row index into the BBI DataFrame for the current date.
                 Must be the correct integer position (not -1).
        
        Returns:
            dict with keys: action ('hold', 'stop_loss', 'profit_take', 'trend_exit'),
                           shares_to_sell, reason
        """
        if ts_code not in self.positions:
            return {"action": "none", "shares_to_sell": 0, "reason": "no_position"}
        
        pos = self.positions[ts_code]
        pnl_pct = (current_price / pos["entry_price"] - 1.0)
        
        # Update highest price
        if current_price > pos["highest_price"]:
            pos["highest_price"] = current_price
        
        # 1. Check stop-loss
        if pos["entry_above_bbi"]:
            # Case B: two consecutive closes below BBI
            if idx >= 0 and idx < len(current_bbi_df):
                two_day_below = bool(current_bbi_df["two_day_below_bbi"].iloc[idx])
            else:
                two_day_below = False
            
            if two_day_below:
                return {
                    "action": "stop_loss",
                    "shares_to_sell": pos["shares"],
                    "reason": f"bbi_exit (PnL: {pnl_pct*100:.1f}%)",
                }
        else:
            # Case A: -3% from entry
            if pnl_pct <= -0.03:
                return {
                    "action": "stop_loss",
                    "shares_to_sell": pos["shares"],
                    "reason": f"stop_3pct (PnL: {pnl_pct*100:.1f}%)",
                }
        
        # 2. Check profit-taking
        for threshold, sell_pct in self.profit_rules:
            if threshold not in pos["profit_taken_levels"] and pnl_pct >= threshold:
                pos["profit_taken_levels"].add(threshold)
                shares_to_sell = int(pos["shares"] * sell_pct)
                pos["shares"] -= shares_to_sell
                
                if pos["shares"] <= 0:
                    return {
                        "action": "profit_take_close",
                        "shares_to_sell": shares_to_sell,
                        "reason": f"profit_take_{threshold*100:.0f}pct_all (PnL: {pnl_pct*100:.1f}%)",
                    }
                
                return {
                    "action": "profit_take",
                    "shares_to_sell": shares_to_sell,
                    "reason": f"profit_take_{threshold*100:.0f}pct_{sell_pct*100:.0f}pct (PnL: {pnl_pct*100:.1f}%)",
                }
        
        return {"action": "hold", "shares_to_sell": 0, "reason": ""}
    
    def close_position(self, ts_code: str):
        """Remove a fully closed position."""
        self.positions.pop(ts_code, None)
    
    def get_position_count(self) -> int:
        """Number of open positions."""
        return len(self.positions)
    
    def get_total_exposure(self, current_prices: dict) -> float:
        """Total market value of all open positions."""
        total = 0.0
        for ts_code, pos in self.positions.items():
            price = current_prices.get(ts_code, pos["entry_price"])
            total += price * pos["shares"]
        return total
