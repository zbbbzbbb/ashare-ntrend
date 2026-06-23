"""
Market Timing — MACD-based bull/bear regime detection using CSI300.

Determines whether the market environment supports new positions.
When bearish: NO new positions, existing may be reduced/exited.
When bullish: new positions allowed with full strategy execution.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from signals.macd import compute_macd, full_bullish_regime, macd_bearish, macd_zero_cross_up


class MarketTiming:
    """MACD-based market timing using a benchmark index (default: CSI300).
    
    Bullish Regime (allow new positions):
      1. DIF > DEA
      2. MACD Histogram Expanding
      3. DIF Rising
      Priority: DIF moving from below zero toward zero ("0下到0上")
    
    Bearish Regime (no new positions):
      1. DIF < DEA
    
    Neutral: neither fully bullish nor bearish (hold existing, no new entries).
    """
    
    def __init__(self, benchmark_code: str = "000300.SH", 
                 fast: int = 12, slow: int = 26, signal: int = 9):
        self.benchmark_code = benchmark_code
        self.fast = fast
        self.slow = slow
        self.signal = signal
        
        # State storage
        self.macd_df = None
        self.regime = None  # Series: 'bullish', 'bearish', 'neutral'
        self.priority_setup = None  # Series: True when "0下到0上" setup
        self.close = None
    
    def fit(self, close: pd.Series):
        """Fit market timing model on benchmark index price series.
        
        Args:
            close: Series of benchmark closing prices (adjusted)
        """
        self.close = close
        
        # Compute MACD
        self.macd_df = compute_macd(close, self.fast, self.slow, self.signal)
        
        # Classify regime
        bullish = full_bullish_regime(self.macd_df)
        bearish = macd_bearish(self.macd_df)
        
        self.regime = pd.Series("neutral", index=close.index)
        self.regime[bullish] = "bullish"
        self.regime[bearish] = "bearish"
        
        # Priority setup: DIF approaching zero from below
        self.priority_setup = macd_zero_cross_up(self.macd_df)
        
        return self
    
    def is_bullish(self, idx: int = -1) -> bool:
        """Check if market is bullish at given index."""
        if self.regime is None:
            return False
        return self.regime.iloc[idx] == "bullish"
    
    def is_bearish(self, idx: int = -1) -> bool:
        """Check if market is bearish at given index."""
        if self.regime is None:
            return True  # Conservative: assume bearish when unknown
        return self.regime.iloc[idx] == "bearish"
    
    def is_priority_setup(self, idx: int = -1) -> bool:
        """Check for highest-priority '0下到0上' setup."""
        if self.priority_setup is None:
            return False
        return self.priority_setup.iloc[idx]
    
    def can_enter(self, idx: int = -1) -> bool:
        """Can new positions be entered? (bullish regime)"""
        return self.is_bullish(idx)
    
    def get_regime_stats(self) -> dict:
        """Get regime distribution statistics."""
        if self.regime is None:
            return {}
        
        counts = self.regime.value_counts()
        total = len(self.regime)
        
        return {
            "total_days": total,
            "bullish_days": int(counts.get("bullish", 0)),
            "bearish_days": int(counts.get("bearish", 0)),
            "neutral_days": int(counts.get("neutral", 0)),
            "bullish_pct": round(100 * counts.get("bullish", 0) / total, 1),
            "bearish_pct": round(100 * counts.get("bearish", 0) / total, 1),
            "priority_setups": int(self.priority_setup.sum()) if self.priority_setup is not None else 0,
        }
    
    def get_regime_returns(self) -> pd.DataFrame:
        """Compute benchmark returns conditioned on regime."""
        if self.close is None:
            return pd.DataFrame()
        
        ret = self.close.pct_change()
        
        results = {"daily_ret": ret}
        for regime in ["bullish", "bearish", "neutral"]:
            mask = self.regime == regime
            results[f"{regime}_ret"] = ret.where(mask, 0)
        
        return pd.DataFrame(results, index=self.close.index)


def evaluate_timing_robustness(
    close: pd.Series,
    fast_params: list = [10, 12, 16],
    slow_params: list = [24, 26, 30],
) -> pd.DataFrame:
    """Evaluate market timing robustness across MACD parameter variations.
    
    Returns DataFrame with regime stats for each parameter combination.
    """
    results = []
    
    for fast in fast_params:
        for slow in slow_params:
            if slow <= fast:
                continue
            
            mt = MarketTiming(fast=fast, slow=slow)
            mt.fit(close)
            stats = mt.get_regime_stats()
            stats["fast"] = fast
            stats["slow"] = slow
            results.append(stats)
    
    return pd.DataFrame(results)
