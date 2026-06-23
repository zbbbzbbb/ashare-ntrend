"""
Walk-Forward Validation with Purged Embargo.

Splits data into sequential training/validation windows with:
  - Purge: remove overlap between train and test to prevent data leakage
  - Embargo: gap between train end and test start
  - Rolling or anchored walk-forward

Each window: train on in-sample, evaluate on out-of-sample.
Aggregate OOS performance across all windows.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Callable
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def purge_embargo_split(
    dates: List[str],
    n_splits: int = 5,
    train_pct: float = 0.6,
    purge_pct: float = 0.05,
    embargo_pct: float = 0.02,
) -> List[Dict]:
    """Generate walk-forward train/test splits with purging and embargo.
    
    Args:
        dates: Sorted list of trading dates
        n_splits: Number of walk-forward windows
        train_pct: Fraction of data for training in each split
        purge_pct: Fraction to purge between train and test
        embargo_pct: Fraction to embargo after purge
    
    Returns:
        List of dicts with 'train_start', 'train_end', 'test_start', 'test_end'
    """
    n = len(dates)
    splits = []
    
    # Total window size
    window_size = int(n * (train_pct + purge_pct + embargo_pct))
    step_size = max(1, (n - window_size) // (n_splits - 1)) if n_splits > 1 else 0
    
    for i in range(n_splits):
        train_end_idx = min(int(n * train_pct) + i * step_size, n - 1)
        
        purge_size = max(1, int(n * purge_pct))
        embargo_size = max(1, int(n * embargo_pct))
        
        test_start_idx = min(train_end_idx + purge_size + embargo_size, n - 1)
        test_end_idx = min(test_start_idx + window_size - purge_size - embargo_size, n)
        
        if test_end_idx <= test_start_idx:
            break
        
        splits.append({
            "split": i,
            "train_start": dates[0],
            "train_end": dates[train_end_idx],
            "test_start": dates[test_start_idx],
            "test_end": dates[test_end_idx],
        })
    
    return splits


def walk_forward_validate(
    backtest_fn: Callable,
    dates: List[str],
    n_splits: int = 5,
    train_pct: float = 0.60,
    purge_pct: float = 0.05,
    embargo_pct: float = 0.02,
    base_params: Dict = None,
) -> Dict:
    """Run walk-forward validation.
    
    Args:
        backtest_fn: Function(params) → dict with performance metrics
        dates: Trading calendar dates
        n_splits: Number of windows
        train_pct: Training fraction
        purge_pct: Purge fraction
        embargo_pct: Embargo fraction
        base_params: Base backtest parameters
    
    Returns:
        Dict with OOS metrics, IS metrics, and stability stats
    """
    if base_params is None:
        base_params = {}
    
    splits = purge_embargo_split(dates, n_splits, train_pct, purge_pct, embargo_pct)
    
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD VALIDATION ({len(splits)} splits)")
    print(f"Purge: {purge_pct:.0%} | Embargo: {embargo_pct:.0%}")
    print(f"{'='*60}")
    
    oos_results = []
    is_results = []
    
    for split in splits:
        print(f"\nSplit {split['split']+1}/{len(splits)}")
        print(f"  Train: {split['train_start']} → {split['train_end']}")
        print(f"  Test:  {split['test_start']} → {split['test_end']}")
        
        # In-sample
        is_params = {**base_params, "start_date": split["train_start"], 
                     "end_date": split["train_end"]}
        is_result = backtest_fn(is_params)
        is_results.append(is_result)
        
        # Out-of-sample
        oos_params = {**base_params, "start_date": split["test_start"],
                      "end_date": split["test_end"]}
        oos_result = backtest_fn(oos_params)
        oos_results.append(oos_result)
        
        print(f"  IS  CAGR: {is_result.get('cagr_pct', 0):.2f}%")
        print(f"  OOS CAGR: {oos_result.get('cagr_pct', 0):.2f}%")
    
    # Aggregate OOS
    oos_cagrs = [r.get("cagr_pct", 0) for r in oos_results]
    is_cagrs = [r.get("is_cagr_pct", 0) for r in is_results]
    oos_dds = [r.get("max_drawdown_pct", 0) for r in oos_results]
    
    # Stability: correlation between IS and OOS
    is_oos_corr = np.corrcoef(is_cagrs, oos_cagrs)[0, 1] if len(is_cagrs) > 1 else 0
    
    summary = {
        "n_splits": len(splits),
        "oos_cagr_mean": np.mean(oos_cagrs),
        "oos_cagr_std": np.std(oos_cagrs),
        "oos_cagr_min": np.min(oos_cagrs),
        "oos_cagr_max": np.max(oos_cagrs),
        "oos_dd_mean": np.mean(oos_dds),
        "is_cagr_mean": np.mean(is_cagrs),
        "is_oos_correlation": is_oos_corr,
        "cagr_stability": 1 - (np.std(oos_cagrs) / abs(np.mean(oos_cagrs))) if np.mean(oos_cagrs) != 0 else 0,
        "oos_results": oos_results,
        "is_results": is_results,
        "splits": splits,
    }
    
    print(f"\n{'='*60}")
    print("WALK-FORWARD SUMMARY")
    print(f"  OOS CAGR: {summary['oos_cagr_mean']:.2f}% ± {summary['oos_cagr_std']:.2f}%")
    print(f"  OOS CAGR Range: [{summary['oos_cagr_min']:.2f}%, {summary['oos_cagr_max']:.2f}%]")
    print(f"  OOS MaxDD Mean: {summary['oos_dd_mean']:.2f}%")
    print(f"  IS/OOS Correlation: {summary['is_oos_correlation']:.3f}")
    print(f"  CAGR Stability: {summary['cagr_stability']:.3f}")
    print(f"{'='*60}")
    
    return summary
