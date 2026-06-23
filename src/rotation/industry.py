"""
Industry Rotation — 88-series (三级行业) industry selection.

Selects industries meeting all four criteria:
  1. MACD Bullish (DIF > DEA)
  2. Relative Strength Leadership (top-ranked over 20 trading days)
  3. Volume Expansion (turnover > 60-day average)
  4. N-Pattern Structure (continuation pattern)

Priority: industries attracting incremental capital.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from signals.macd import compute_macd, macd_bullish
from signals.n_pattern import detect_n_pattern
from signals.relative_strength import relative_strength, rs_rank


class IndustryRotator:
    """Selects top industries for rotation based on trend + momentum criteria."""
    
    def __init__(
        self,
        top_n: int = 5,
        rs_period: int = 20,
        volume_period: int = 60,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        n_leg1_start: int = 20,
        n_leg1_end: int = 10,
        n_pullback_end: int = 5,
    ):
        self.top_n = top_n
        self.rs_period = rs_period
        self.volume_period = volume_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.n_leg1_start = n_leg1_start
        self.n_leg1_end = n_leg1_end
        self.n_pullback_end = n_pullback_end
    
    def score_industries(
        self, 
        industry_data: dict,  # {index_code: pd.DataFrame with OHLCV}
        benchmark_close: pd.Series = None,
        date_idx: int = -1,
    ) -> pd.DataFrame:
        """Score all industries and return ranked DataFrame.
        
        Scoring criteria (each contributes 0-1):
          1. MACD bullish: 0 or 1
          2. RS percentile rank: 0-1
          3. Volume expansion ratio (capped): 0-1
          4. N-pattern quality: 0-1
        
        Composite = equally weighted average.
        """
        scores = []
        
        for idx_code, df in industry_data.items():
            if len(df) <= date_idx or len(df) < 60:
                continue
            
            close = df["close"]
            vol = df["vol"] if "vol" in df.columns else pd.Series(1, index=close.index)
            
            # 1. MACD bullish
            macd_df = compute_macd(close, self.macd_fast, self.macd_slow, self.macd_signal)
            macd_ok = float(macd_bullish(macd_df).iloc[date_idx])
            
            # 2. Relative Strength (vs benchmark or cross-sectional)
            if benchmark_close is not None:
                rs = relative_strength(close, benchmark_close, self.rs_period)
            else:
                rs = relative_strength(close, period=self.rs_period)
            rs_val = rs.iloc[date_idx]
            
            # 3. Volume expansion
            avg_vol = vol.rolling(window=self.volume_period).mean()
            vol_ratio = (vol.iloc[date_idx] / avg_vol.iloc[date_idx]) if avg_vol.iloc[date_idx] > 0 else 0
            vol_score = min(vol_ratio / 2.0, 1.0)  # Cap at 2x average
            
            # 4. N-pattern
            n_df = detect_n_pattern(
                close, 
                leg1_start=self.n_leg1_start,
                leg1_end=self.n_leg1_end,
                pullback_end=self.n_pullback_end,
            )
            n_ok = float(n_df["is_n_pattern"].iloc[date_idx])
            n_quality = n_df["pattern_quality"].iloc[date_idx]
            
            composite = (macd_ok + vol_score + n_quality) / 3.0
            
            scores.append({
                "index_code": idx_code,
                "macd_bullish": macd_ok,
                "rs_value": rs_val,
                "volume_ratio": vol_ratio,
                "volume_score": vol_score,
                "n_pattern": n_ok,
                "n_quality": n_quality,
                "composite_score": composite,
            })
        
        if not scores:
            return pd.DataFrame()
        
        result = pd.DataFrame(scores)
        
        # RS rank (cross-sectional among all industries)
        result["rs_rank"] = result["rs_value"].rank(ascending=False, pct=True)
        
        # Final composite with RS
        result["final_score"] = (
            result["macd_bullish"] * 0.25 +
            result["rs_rank"] * 0.35 +
            result["volume_score"] * 0.20 +
            result["n_quality"] * 0.20
        )
        
        return result.sort_values("final_score", ascending=False)
    
    def select_industries(
        self,
        industry_data: dict,
        benchmark_close: pd.Series = None,
        date_idx: int = -1,
    ) -> list:
        """Select top-N industries meeting all basic criteria.
        
        Returns list of selected index_codes.
        """
        scored = self.score_industries(industry_data, benchmark_close, date_idx)
        
        if len(scored) == 0:
            return []
        
        # Filter: must have MACD bullish
        qualified = scored[scored["macd_bullish"] > 0]
        
        # If too few MACD-bullish industries, relax to top by composite
        if len(qualified) < self.top_n:
            qualified = scored.head(self.top_n)
        
        selected = qualified.head(self.top_n)["index_code"].tolist()
        return selected
    
    def get_concentration(self, scored_df: pd.DataFrame, top_n: int = 5) -> float:
        """Measure industry concentration: sum of top-N scores / total sum."""
        if len(scored_df) == 0:
            return 0.0
        total = scored_df["final_score"].sum()
        if total == 0:
            return 0.0
        top_sum = scored_df.head(top_n)["final_score"].sum()
        return top_sum / total


def build_industry_data(loader, index_codes: list, start: str, end: str) -> dict:
    """Build industry price data dict from the data lake.
    
    Args:
        loader: Data loader module
        index_codes: List of industry index codes
        start, end: Date range
    
    Returns:
        {index_code: DataFrame with trade_date, open, high, low, close, vol}
    """
    from ..data.loader import get_index_daily
    
    industry_data = {}
    for idx_code in index_codes:
        df = get_index_daily(idx_code, start, end)
        if len(df) > 0:
            df = df.sort_values("trade_date").reset_index(drop=True)
            industry_data[idx_code] = df
    
    return industry_data
