"""
MACD Indicator — core trend-following signal.

Computes DIF (fast EMA - slow EMA), DEA (signal line), 
and histogram (DIF - DEA) for regime detection and entry confirmation.

Key MACD states for this strategy:
  - DIF > DEA        → bullish bias
  - DIF < DEA        → bearish bias  
  - Histogram expanding → trend strengthening
  - DIF crossing above zero → "0下到0上" (highest priority setup)
"""

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Compute MACD for a price series.
    
    Args:
        close: Series of closing prices (adjusted)
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)
    
    Returns:
        DataFrame with columns: dif, dea, histogram, dif_rising, histogram_expanding
    """
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    
    dif = ema_fast - ema_slow
    dea = ema(dif, signal)
    histogram = (dif - dea) * 2  # Standard MACD histogram = 2 × (DIF - DEA)
    
    # Derived states
    dif_rising = dif > dif.shift(1)
    histogram_expanding = histogram.abs() > histogram.abs().shift(1)
    
    return pd.DataFrame({
        "dif": dif,
        "dea": dea,
        "histogram": histogram,
        "dif_rising": dif_rising,
        "histogram_expanding": histogram_expanding,
    }, index=close.index)


def macd_bullish(macd_df: pd.DataFrame) -> pd.Series:
    """Check if MACD is in bullish regime: DIF > DEA."""
    return macd_df["dif"] > macd_df["dea"]


def macd_bearish(macd_df: pd.DataFrame) -> pd.Series:
    """Check if MACD is in bearish regime: DIF < DEA."""
    return macd_df["dif"] < macd_df["dea"]


def macd_zero_cross_up(macd_df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Detect DIF crossing from below zero toward zero (0下到0上).
    
    Returns True when DIF was negative recently and is now approaching zero
    from below (rising toward zero). This is the 'highest priority' setup.
    """
    dif = macd_df["dif"]
    dea = macd_df["dea"]
    
    # DIF is currently negative but rising
    approaching_zero = (dif < 0) & dif_rising(dif)
    
    # And was below DEA recently but now above (or close to crossing)
    dea_cross_imminent = (dif > dif.shift(lookback)) & (dif < 0)
    
    return approaching_zero & dea_cross_imminent


def dif_rising(dif: pd.Series, periods: int = 1) -> pd.Series:
    """Check if DIF is rising (current > previous)."""
    return dif > dif.shift(periods)


def histogram_expanding(macd_df: pd.DataFrame) -> pd.Series:
    """Check if MACD histogram is expanding (in absolute value)."""
    return macd_df["histogram"].abs() > macd_df["histogram"].abs().shift(1)


def full_bullish_regime(macd_df: pd.DataFrame) -> pd.Series:
    """Full bullish regime: DIF > DEA, histogram expanding, DIF rising."""
    return (
        macd_bullish(macd_df) & 
        macd_df["histogram_expanding"] & 
        macd_df["dif_rising"]
    )


class MACDState:
    """MACD state classifier for market timing.
    
    Bullish (allow positions):
      1. DIF > DEA
      2. Histogram expanding
      3. DIF rising
      Priority: DIF moving from below zero toward zero
    
    Bearish (no new positions):
      1. DIF < DEA
    """
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def fit(self, close: pd.Series):
        """Compute MACD and classify states."""
        self.macd = compute_macd(close, self.fast, self.slow, self.signal)
        self.bullish = full_bullish_regime(self.macd)
        self.bearish = macd_bearish(self.macd)
        self.neutral = ~(self.bullish | self.bearish)
        self.zero_cross_priority = macd_zero_cross_up(self.macd)
        return self
    
    def get_state(self, idx: int = -1) -> str:
        """Get current MACD state: 'bullish', 'bearish', or 'neutral'."""
        if self.bullish.iloc[idx]:
            return "bullish"
        elif self.bearish.iloc[idx]:
            return "bearish"
        return "neutral"
    
    def can_enter(self, idx: int = -1) -> bool:
        """Check if new entries are allowed."""
        return self.bullish.iloc[idx]
