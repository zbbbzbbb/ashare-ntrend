"""
Performance Metrics — comprehensive strategy evaluation.

Computes:
  - CAGR (Compound Annual Growth Rate)
  - Max Drawdown & Calmar Ratio
  - Sharpe Ratio (annualized)
  - Sortino Ratio
  - Profit Factor
  - Win Rate & Avg Win/Loss
  - Turnover
  - Regime-conditional returns
"""

import numpy as np
import pandas as pd
from typing import List, Dict


def compute_returns(nav_series: pd.Series) -> pd.Series:
    """Compute daily returns from NAV series."""
    return nav_series.pct_change().dropna()


def cagr(nav_series: pd.Series) -> float:
    """Compound Annual Growth Rate."""
    if len(nav_series) < 2:
        return 0.0
    total_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1
    years = len(nav_series) / 252
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1


def max_drawdown(nav_series: pd.Series) -> float:
    """Maximum drawdown (negative value, e.g. -0.25 = -25%)."""
    peak = nav_series.cummax()
    dd = (nav_series - peak) / peak
    return float(dd.min())


def calmar_ratio(nav_series: pd.Series) -> float:
    """Calmar ratio = CAGR / |Max DD|."""
    c = cagr(nav_series)
    mdd = abs(max_drawdown(nav_series))
    return c / mdd if mdd > 0 else 0.0


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.02) -> float:
    """Annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / 252
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / excess.std())


def sortino_ratio(returns: pd.Series, risk_free: float = 0.02) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / 252
    downside = excess[excess < 0]
    if len(downside) < 2 or downside.std() == 0:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / downside.std())


def profit_factor(trades: List[Dict]) -> float:
    """Gross profit / Gross loss from trade list."""
    if not trades:
        return 0.0
    
    sells = [t for t in trades if t.get("action") == "sell"]
    if not sells:
        return 0.0
    
    gross_profit = sum(t["pnl"] for t in sells if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in sells if t["pnl"] < 0))
    
    return gross_profit / gross_loss if gross_loss > 0 else float("inf")


def win_rate(trades: List[Dict]) -> float:
    """Fraction of winning trades."""
    sells = [t for t in trades if t.get("action") == "sell"]
    if not sells:
        return 0.0
    return sum(1 for t in sells if t["pnl"] > 0) / len(sells)


def avg_trade_metrics(trades: List[Dict]) -> Dict:
    """Average win, average loss, win/loss ratio."""
    sells = [t for t in trades if t.get("action") == "sell"]
    if not sells:
        return {"avg_win": 0, "avg_loss": 0, "win_loss_ratio": 0}
    
    wins = [t["pnl"] for t in sells if t["pnl"] > 0]
    losses = [t["pnl"] for t in sells if t["pnl"] < 0]
    
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    
    return {
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": avg_win / avg_loss if avg_loss > 0 else 0,
    }


def turnover(trades: List[Dict], avg_nav: float) -> float:
    """Annual turnover rate."""
    if not trades or avg_nav <= 0:
        return 0.0
    
    total_traded = sum(
        abs(t["price"] * t["shares"]) for t in trades
        if t.get("action") in ("buy", "sell")
    )
    
    return total_traded / avg_nav


def regime_returns(nav_df: pd.DataFrame, regime_series: pd.Series) -> Dict:
    """Compute returns conditioned on market regime."""
    if nav_df is None or regime_series is None:
        return {}
    
    returns = compute_returns(nav_df["nav"])
    
    # Align indices
    common_idx = returns.index.intersection(regime_series.index)
    
    results = {}
    for regime in ["bullish", "bearish", "neutral"]:
        mask = regime_series.loc[common_idx] == regime
        regime_ret = returns.loc[common_idx][mask]
        
        if len(regime_ret) > 0:
            results[f"{regime}_cagr"] = (1 + regime_ret.mean()) ** 252 - 1
            results[f"{regime}_sharpe"] = sharpe_ratio(regime_ret)
            results[f"{regime}_days"] = len(regime_ret)
            results[f"{regime}_pct"] = len(regime_ret) / len(returns) * 100
        else:
            results[f"{regime}_cagr"] = 0
            results[f"{regime}_sharpe"] = 0
            results[f"{regime}_days"] = 0
            results[f"{regime}_pct"] = 0
    
    return results


def full_metrics(nav_df: pd.DataFrame, trades: List[Dict], 
                 regime_series: pd.Series = None) -> Dict:
    """Compute all performance metrics.
    
    Args:
        nav_df: DataFrame with 'date' and 'nav' columns (or Series of NAV)
        trades: List of trade dicts
        regime_series: Market regime classification (optional)
    
    Returns:
        Dict of all metrics
    """
    if isinstance(nav_df, pd.DataFrame) and "nav" in nav_df.columns:
        nav_series = nav_df.set_index("date")["nav"]
    elif isinstance(nav_df, pd.Series):
        nav_series = nav_df
    else:
        return {}
    
    returns = compute_returns(nav_series)
    
    metrics = {
        "cagr": cagr(nav_series),
        "max_drawdown": max_drawdown(nav_series),
        "calmar_ratio": calmar_ratio(nav_series),
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "profit_factor": profit_factor(trades),
        "win_rate": win_rate(trades),
        **avg_trade_metrics(trades),
        "total_return": nav_series.iloc[-1] / nav_series.iloc[0] - 1,
        "total_trades": len(trades),
        "num_buys": len([t for t in trades if t.get("action") == "buy"]),
        "num_sells": len([t for t in trades if t.get("action") == "sell"]),
        "years": len(nav_series) / 252,
    }
    
    if regime_series is not None:
        metrics.update(regime_returns(
            pd.DataFrame({"date": nav_series.index, "nav": nav_series.values}),
            regime_series
        ))
    
    return metrics
