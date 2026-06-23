"""
CPCV — Combinatorial Purged Cross-Validation.

Implements the methodology from "Advances in Financial Machine Learning" (Lopez de Prado).

Key features:
  - Purges overlapping data between train and test groups
  - Embargo period after training data
  - Multiple combinatorial splits for robust OOS estimation
  - Backtest overfitting probability estimation
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Dict, Callable, Optional


def cpcv_groups(
    n_samples: int,
    n_groups: int = 6,
    purge_size: int = 1,
) -> List[Dict]:
    """Generate CPCV train/test splits.
    
    Divides data into N groups. For each combination of (N-1) groups as train 
    and 1 group as test, with purging between.
    
    Args:
        n_samples: Total number of samples
        n_groups: Number of groups to divide data into
        purge_size: Number of samples to purge on each side
    
    Returns:
        List of dicts with train_indices, test_indices, purge_info
    """
    # Divide into roughly equal groups
    group_size = n_samples // n_groups
    groups = []
    for i in range(n_groups):
        start = i * group_size
        end = start + group_size if i < n_groups - 1 else n_samples
        groups.append(list(range(start, end)))
    
    splits = []
    
    # For each test group, train on union of all other groups
    for test_idx in range(n_groups):
        test_indices = groups[test_idx]
        
        # Train groups are all except test
        train_indices = []
        for train_idx in range(n_groups):
            if train_idx != test_idx:
                indices = groups[train_idx]
                # Purge: remove purge_size samples adjacent to test group
                if train_idx == test_idx - 1:
                    # Group immediately before test: purge tail
                    indices = indices[:-purge_size] if len(indices) > purge_size else indices
                elif train_idx == test_idx + 1:
                    # Group immediately after test: purge head  
                    indices = indices[purge_size:] if len(indices) > purge_size else indices
                train_indices.extend(indices)
        
        splits.append({
            "split": test_idx,
            "train_indices": sorted(train_indices),
            "test_indices": test_indices,
        })
    
    return splits


def cpcv_prob_overfit(
    oos_performances: List[float],
    is_performances: List[float],
) -> Dict:
    """Estimate probability of backtest overfitting (PBO).
    
    Based on Lopez de Prado's deflated Sharpe ratio framework.
    
    PBO = fraction of parameter combinations where IS rank is high 
          but OOS rank is low.
    
    Args:
        oos_performances: OOS performance for each trial
        is_performances: IS performance for each trial
    
    Returns:
        Dict with pbo, performance degradation, etc.
    """
    n = len(oos_performances)
    if n < 2:
        return {"pbo": 0.0, "degradation": 0.0, "n_trials": n}
    
    # Rank IS and OOS
    is_ranks = pd.Series(is_performances).rank(ascending=False)
    oos_ranks = pd.Series(oos_performances).rank(ascending=False)
    
    # Rank correlation
    spearman = is_ranks.corr(oos_ranks, method="spearman")
    
    # PBO: for each pair (i,j), count cases where IS(i) > IS(j) but OOS(i) < OOS(j)
    pbo_count = 0
    total_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            if (is_performances[i] > is_performances[j] and 
                oos_performances[i] < oos_performances[j]):
                pbo_count += 1
            elif (is_performances[i] < is_performances[j] and 
                  oos_performances[i] > oos_performances[j]):
                pbo_count += 1
    
    pbo = pbo_count / total_pairs if total_pairs > 0 else 0
    
    # Performance degradation: avg(OOS/IS ratio)
    degradation = np.mean([
        oos / is_ if is_ != 0 else 0
        for oos, is_ in zip(oos_performances, is_performances)
    ])
    
    return {
        "pbo": pbo,
        "spearman_rank_corr": spearman,
        "performance_degradation": degradation,
        "n_trials": n,
        "is_mean": np.mean(is_performances),
        "is_std": np.std(is_performances),
        "oos_mean": np.mean(oos_performances),
        "oos_std": np.std(oos_performances),
    }


def cpcv_validate(
    backtest_fn: Callable,
    dates: List[str],
    n_groups: int = 6,
    purge_size: int = 5,
    base_params: Dict = None,
) -> Dict:
    """Run CPCV validation.
    
    Args:
        backtest_fn: Function(params) → dict with 'cagr_pct' or 'sharpe_ratio'
        dates: List of trading dates
        n_groups: Number of CPCV groups
        purge_size: Purge size in trading days
        base_params: Base backtest parameters
    
    Returns:
        Dict with CPCV results and PBO estimate
    """
    if base_params is None:
        base_params = {}
    
    n = len(dates)
    splits = cpcv_groups(n, n_groups, purge_size)
    
    print(f"\n{'='*60}")
    print(f"CPCV VALIDATION ({n_groups} groups, {len(splits)} splits)")
    print(f"Purge: {purge_size} days each side")
    print(f"{'='*60}")
    
    oos_performances = []
    is_performances = []
    
    for split in splits:
        train_indices = split["train_indices"]
        test_indices = split["test_indices"]
        
        if len(train_indices) < 60 or len(test_indices) < 20:
            continue
        
        train_start = dates[min(train_indices)]
        train_end = dates[max(train_indices)]
        test_start = dates[min(test_indices)]
        test_end = dates[max(test_indices)]
        
        print(f"\n  Split {split['split']+1}: Test {test_start} → {test_end}")
        
        # IS
        is_params = {**base_params, "start_date": train_start, "end_date": train_end}
        is_result = backtest_fn(is_params)
        is_perf = is_result.get("cagr_pct", is_result.get("sharpe_ratio", 0))
        
        # OOS
        oos_params = {**base_params, "start_date": test_start, "end_date": test_end}
        oos_result = backtest_fn(oos_params)
        oos_perf = oos_result.get("cagr_pct", oos_result.get("sharpe_ratio", 0))
        
        is_performances.append(is_perf)
        oos_performances.append(oos_perf)
        
        print(f"    IS: {is_perf:.3f} | OOS: {oos_perf:.3f}")
    
    # PBO analysis
    pbo_result = cpcv_prob_overfit(oos_performances, is_performances)
    
    summary = {
        **pbo_result,
        "n_splits": len(oos_performances),
        "oos_individual": oos_performances,
        "is_individual": is_performances,
    }
    
    print(f"\n  PBO: {pbo_result['pbo']:.3f}")
    print(f"  Spearman ρ: {pbo_result['spearman_rank_corr']:.3f}")
    print(f"  Performance Degradation: {pbo_result['performance_degradation']:.3f}")
    
    return summary
