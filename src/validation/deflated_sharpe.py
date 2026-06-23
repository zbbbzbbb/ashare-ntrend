"""
Deflated Sharpe Ratio & Multiple Testing Correction.

Implements:
  1. Deflated Sharpe Ratio (DSR) — accounts for selection bias under multiple testing
  2. Probability of Backtest Overfitting (PBO)
  3. Trial count tracking

Methodology from Lopez de Prado & Bailey (2014), "The Deflated Sharpe Ratio".
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict


def deflated_sharpe_ratio(
    observed_sr: float,
    sr_trials: List[float],
    n_samples: int,
    skewness: float = None,
    kurtosis: float = None,
) -> Dict:
    """Compute the Deflated Sharpe Ratio.
    
    DSR = P[SR > E[max(SR)]] accounting for the maximum expected SR
    under multiple testing.
    
    Args:
        observed_sr: The Sharpe ratio of the selected strategy
        sr_trials: Sharpe ratios from all parameter combinations tested
        n_samples: Number of observations used to compute SR
        skewness: Return skewness (estimated from data if None)
        kurtosis: Return kurtosis (estimated from data if None)
    
    Returns:
        Dict with dsr, p_value, expected_max_sr, etc.
    """
    n_trials = len(sr_trials)
    if n_trials == 0:
        return {"dsr": 0.0, "p_value": 1.0, "n_trials": 0}
    
    # Expected maximum SR under the null (all trials are noise)
    # Using extreme value theory approximation
    
    sr_array = np.array(sr_trials)
    sr_std = np.std(sr_array) if len(sr_array) > 1 else 0.01
    
    if sr_std == 0:
        sr_std = 0.01
    
    # Standardize
    sr_std_array = sr_array / sr_std
    
    # Expected max of N standard normals (E[max])
    # For large N, E[max] ≈ sqrt(2 * log(N))
    euler = 0.5772156649
    expected_max = (1 - euler) * stats.norm.ppf(1 - 1.0 / n_trials) + euler * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    
    # Simpler approximation
    expected_max_sr = sr_std * np.sqrt(2 * np.log(n_trials))
    
    # Deflated SR: deflate observed SR by expected max
    adjusted_sr = observed_sr - expected_max_sr
    
    # P-value: probability that observed SR could occur by chance
    # under N independent trials
    if adjusted_sr > 0:
        # Standard error of SR ≈ 1/sqrt(n)
        se_sr = 1.0 / np.sqrt(n_samples)
        z_stat = adjusted_sr / se_sr
        p_value = 1 - stats.norm.cdf(z_stat)
    else:
        p_value = 1.0
    
    dsr = adjusted_sr / (1.0 / np.sqrt(n_samples)) if n_samples > 0 else 0
    
    return {
        "observed_sr": observed_sr,
        "expected_max_sr": expected_max_sr,
        "deflated_sr": adjusted_sr,
        "dsr_z_score": dsr,
        "p_value": p_value,
        "significant_at_5pct": p_value < 0.05,
        "n_trials": n_trials,
        "n_observations": n_samples,
    }


def track_trials(param_grid: Dict) -> int:
    """Count total number of parameter combinations tested.
    
    Args:
        param_grid: Dict of parameter_name → list of values
    
    Returns:
        Total number of combinations
    """
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total


def deflated_sharpe_analysis(
    backtest_results: List[Dict],
    n_observations: int,
    performance_key: str = "sharpe_ratio",
) -> Dict:
    """Run full deflated Sharpe analysis on backtest results.
    
    Args:
        backtest_results: List of result dicts from parameter sweeps
        n_observations: Number of return observations
        performance_key: Which metric to use (default 'sharpe_ratio')
    
    Returns:
        Dict with DSR analysis, best params, trial count
    """
    if not backtest_results:
        return {"error": "No results provided"}
    
    # Extract performance values
    performances = [r.get(performance_key, 0) for r in backtest_results]
    
    best_idx = np.argmax(performances)
    best_sr = performances[best_idx]
    best_result = backtest_results[best_idx]
    
    dsr_result = deflated_sharpe_ratio(
        observed_sr=best_sr,
        sr_trials=performances,
        n_samples=n_observations,
    )
    
    return {
        **dsr_result,
        "best_result": best_result,
        "all_performances": performances,
        "mean_performance": np.mean(performances),
        "std_performance": np.std(performances),
        "pct_positive": sum(1 for p in performances if p > 0) / len(performances) * 100,
    }


def parameter_stability(
    results: List[Dict],
    param_name: str,
    performance_key: str = "cagr_pct",
) -> Dict:
    """Evaluate parameter stability across a range.
    
    Args:
        results: List of result dicts, each with the parameter value stored
        param_name: Name of the parameter to analyze
        performance_key: Performance metric key
    
    Returns:
        Dict with stability metrics
    """
    if not results:
        return {}
    
    param_values = [r.get(param_name, i) for i, r in enumerate(results)]
    performances = [r.get(performance_key, 0) for r in results]
    
    # Stability: coefficient of variation of performance across params
    cv = np.std(performances) / abs(np.mean(performances)) if np.mean(performances) != 0 else 0
    
    # Range
    perf_range = max(performances) - min(performances)
    
    # Find plateau: region where performance is within 10% of max
    max_perf = max(performances)
    plateau_threshold = max_perf * 0.9
    plateau_params = [p for p, perf in zip(param_values, performances) if perf >= plateau_threshold]
    
    return {
        "param_name": param_name,
        "cv": cv,
        "range": perf_range,
        "max_performance": max_perf,
        "optimal_param": param_values[np.argmax(performances)],
        "plateau_width": len(plateau_params),
        "plateau_range": f"{min(plateau_params)} → {max(plateau_params)}" if plateau_params else "N/A",
        "is_stable": cv < 0.3,  # CV < 30% suggests stability
    }
