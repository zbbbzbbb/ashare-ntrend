"""
N-Pattern Trend Breakout Detection.

The N-pattern is a classic trend continuation structure:
  1. First Leg:   Price rises from T-20 to T-10
  2. Pullback:    Price retraces from T-10 to T-5
  3. Second Leg:  Price breaks above recent high (T-10 peak or T-5 peak)

This pattern captures healthy uptrends where a pullback creates a 
higher-probability entry at the breakout of the second leg.
"""

import numpy as np
import pandas as pd


def detect_n_pattern(
    close: pd.Series,
    high: pd.Series = None,
    leg1_start: int = 20,
    leg1_end: int = 10,
    pullback_end: int = 5,
    min_leg1_return: float = 0.02,
    max_pullback: float = -0.01,
    breakout_pct: float = 0.0,
) -> pd.DataFrame:
    """Detect N-pattern breakouts.
    
    The canonical pattern:
      Leg 1:  close[T-20] → close[T-10] rising (first upswing)
      Pullback: close[T-10] → close[T-5] retracing (correction)
      Breakout: today's close > recent peak (second leg starting)
    
    Args:
        close: Series of adjusted closing prices
        high: Series of adjusted high prices (for breakout confirmation)
        leg1_start: Days back for leg 1 start (default 20)
        leg1_end: Days back for leg 1 end (default 10)
        pullback_end: Days back for pullback end (default 5)
        min_leg1_return: Minimum return for leg 1 to qualify (default 2%)
        max_pullback: Maximum pullback return (negative threshold, default -1%)
        breakout_pct: Minimum breakout above recent high (default 0%)
    
    Returns:
        DataFrame with columns:
          - leg1_return: Return during leg 1
          - pullback_depth: Return during pullback
          - recent_high: Highest price during pattern window
          - is_n_pattern: Boolean, whether pattern is detected
          - breakout_strength: How far above recent high
          - pattern_quality: Composite quality score
    """
    if high is None:
        high = close
    
    n = len(close)
    
    # Compute pattern components
    leg1_start_price = close.shift(leg1_start)
    leg1_end_price = close.shift(leg1_end)
    pullback_end_price = close.shift(pullback_end)
    
    leg1_return = (leg1_end_price / leg1_start_price) - 1.0
    pullback_return = (pullback_end_price / leg1_end_price) - 1.0
    
    # Recent high: max high during the pattern window (leg1_end to pullback_end)
    recent_high = high.rolling(window=leg1_end + 1, min_periods=1).max().shift(1)
    
    # Current breakout
    breakout_strength = (close / recent_high) - 1.0
    
    # Pattern conditions
    leg1_rising = leg1_return > min_leg1_return
    pullback_valid = pullback_return < max_pullback  # Must have pulled back
    breakout = close > recent_high  # Breaking above recent high
    breakout_strong = breakout_strength > breakout_pct
    
    is_pattern = leg1_rising & pullback_valid & breakout & breakout_strong
    
    # Quality score (0-1): combines leg strength, pullback shallowness, breakout strength
    quality = np.zeros(n)
    mask = is_pattern.values
    
    if mask.any():
        # Normalize components to [0, 1]
        leg1_norm = np.clip((leg1_return.values[mask] - min_leg1_return) / 0.10, 0, 1)
        pullback_norm = np.clip(1 - (pullback_return.values[mask] / max_pullback), 0, 1)
        breakout_norm = np.clip(breakout_strength.values[mask] / 0.03, 0, 1)
        
        quality[mask] = (leg1_norm + pullback_norm + breakout_norm) / 3.0
    
    return pd.DataFrame({
        "leg1_return": leg1_return,
        "pullback_depth": pullback_return,
        "recent_high": recent_high,
        "breakout_strength": breakout_strength,
        "is_n_pattern": is_pattern,
        "pattern_quality": quality,
    }, index=close.index)


def detect_n_pattern_variants(
    close: pd.Series,
    high: pd.Series = None,
) -> dict:
    """Test multiple N-pattern parameter variants.
    
    Returns dict of variant_name → DataFrame with is_n_pattern column.
    """
    variants = {
        "standard": (20, 10, 5, 0.02, -0.01),
        "tight": (15, 7, 3, 0.015, -0.005),
        "wide": (30, 15, 7, 0.03, -0.02),
        "shallow_pullback": (20, 10, 5, 0.02, -0.005),
        "deep_pullback": (20, 10, 5, 0.02, -0.03),
    }
    
    results = {}
    for name, (l1_start, l1_end, pb_end, min_l1, max_pb) in variants.items():
        df = detect_n_pattern(
            close, high, 
            leg1_start=l1_start, 
            leg1_end=l1_end,
            pullback_end=pb_end,
            min_leg1_return=min_l1,
            max_pullback=max_pb,
        )
        results[name] = df
    
    return results


def count_patterns(n_pattern_df: pd.DataFrame) -> dict:
    """Count N-pattern occurrences and average quality."""
    mask = n_pattern_df["is_n_pattern"]
    return {
        "total_patterns": int(mask.sum()),
        "pct_days": round(100 * mask.mean(), 2),
        "avg_quality": round(n_pattern_df.loc[mask, "pattern_quality"].mean(), 3) if mask.any() else 0,
        "avg_breakout_strength": round(n_pattern_df.loc[mask, "breakout_strength"].mean() * 100, 2) if mask.any() else 0,
    }
