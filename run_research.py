#!/usr/bin/env python3
"""
A-Share Trend Following Strategy — Main Research Pipeline.

Usage:
  python3 run_research.py build          # Build data lake
  python3 run_research.py audit          # Run data quality audit
  python3 run_research.py backtest       # Run single backtest
  python3 run_research.py validate       # Run full validation suite
  python3 run_research.py final          # Run final holdout evaluation
  python3 run_research.py sweep          # Run parameter sweep
  python3 run_research.py all            # Run complete pipeline

Environment:
  TUSHARE_TOKEN — Tushare Pro API token (required for data download)
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.build_lake import build_catalog
from src.data.audit import run_audit
from src.data.loader import get_connection, get_trade_dates
from src.backtest.engine import BacktestEngine, run_backtest
from src.backtest.costs import CostModel
from src.validation.metrics import full_metrics
from src.validation.walkforward import walk_forward_validate
from src.validation.cpcv import cpcv_validate
from src.validation.deflated_sharpe import deflated_sharpe_analysis, track_trials, parameter_stability

OUTPUT_DIR = Path(__file__).parent / "output"


def cmd_build():
    """Build the data lake from Tushare."""
    build_catalog()


def cmd_audit():
    """Run data quality audit."""
    run_audit()


def cmd_backtest(args):
    """Run a single backtest."""
    params = {
        "start_date": args.start or "20150101",
        "end_date": args.end or "20250101",
        "capital": 1_500_000.0,
        "cost_stress": args.stress or 1.0,
        "top_industries": args.industries or 5,
        "max_positions": args.positions or 10,
        "profit_variant": args.profit_variant or "default",
    }
    
    engine = BacktestEngine(**params)
    results = engine.run()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"backtest_{timestamp}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved: {out_path}")
    
    return results


def cmd_validate(args):
    """Run full validation suite."""
    
    # Get trading dates
    dates_df = get_trade_dates("20150101", "20250101")
    dates = dates_df["cal_date"].tolist()
    
    print(f"\nTotal trading days available: {len(dates):,}")
    
    base_params = {
        "capital": 1_500_000.0,
        "cost_stress": 1.0,
        "top_industries": 5,
        "max_positions": 10,
    }
    
    # 1. Walk-Forward Validation
    print("\n" + "="*70)
    print("PHASE 1: WALK-FORWARD VALIDATION")
    print("="*70)
    
    wf_results = walk_forward_validate(
        backtest_fn=lambda p: BacktestEngine(**p).run(),
        dates=dates,
        n_splits=5,
        train_pct=0.60,
        purge_pct=0.05,
        embargo_pct=0.02,
        base_params=base_params,
    )
    
    # 2. CPCV Validation
    print("\n" + "="*70)
    print("PHASE 2: CPCV VALIDATION")
    print("="*70)
    
    cpcv_results = cpcv_validate(
        backtest_fn=lambda p: BacktestEngine(**p).run(),
        dates=dates,
        n_groups=6,
        purge_size=5,
        base_params=base_params,
    )
    
    # 3. Parameter Stability
    print("\n" + "="*70)
    print("PHASE 3: PARAMETER STABILITY")
    print("="*70)
    
    # Test MACD parameter stability
    param_results = []
    for fast, slow in [(10, 24), (12, 26), (16, 30)]:
        params = {**base_params, "macd_fast": fast, "macd_slow": slow,
                  "start_date": "20150101", "end_date": "20200101"}
        result = BacktestEngine(**params).run()
        result["macd_fast"] = fast
        result["macd_slow"] = slow
        param_results.append(result)
        print(f"  MACD({fast},{slow}): CAGR={result.get('cagr_pct', 0):.2f}%")
    
    stability = parameter_stability(param_results, "macd_fast")
    print(f"  Stability CV: {stability.get('cv', 0):.3f}")
    
    # 4. Trial count
    param_grid = {
        "macd_fast": [10, 12, 16],
        "macd_slow": [24, 26, 30],
        "n_leg1_start": [15, 20, 30],
        "top_industries": [3, 5, 7],
    }
    n_trials = track_trials(param_grid)
    print(f"\n  Total parameter combinations: {n_trials}")
    
    # Compile full validation report
    report = {
        "timestamp": datetime.now().isoformat(),
        "walk_forward": {
            "oos_cagr_mean": wf_results.get("oos_cagr_mean"),
            "oos_cagr_std": wf_results.get("oos_cagr_std"),
            "is_oos_correlation": wf_results.get("is_oos_correlation"),
            "cagr_stability": wf_results.get("cagr_stability"),
        },
        "cpcv": {
            "pbo": cpcv_results.get("pbo"),
            "spearman_rho": cpcv_results.get("spearman_rank_corr"),
            "oos_mean": cpcv_results.get("oos_mean"),
        },
        "parameter_stability": stability,
        "trial_count": n_trials,
    }
    
    out_path = OUTPUT_DIR / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n✅ Validation complete. Report: {out_path}")
    
    return report


def cmd_sweep(args):
    """Run parameter sweep across key dimensions."""
    
    print("\n" + "="*70)
    print("PARAMETER SWEEP")
    print("="*70)
    
    all_results = []
    
    # Sweep configurations
    configs = [
        # (name, params_dict)
        ("Base (12,26)", {"macd_fast": 12, "macd_slow": 26}),
        ("Fast (10,24)", {"macd_fast": 10, "macd_slow": 24}),
        ("Slow (16,30)", {"macd_fast": 16, "macd_slow": 30}),
    ]
    
    # N-pattern variants
    n_configs = [
        ("Standard N", {"n_leg1_start": 20, "n_leg1_end": 10, "n_pullback_end": 5}),
        ("Tight N", {"n_leg1_start": 15, "n_leg1_end": 7, "n_pullback_end": 3}),
        ("Wide N", {"n_leg1_start": 30, "n_leg1_end": 15, "n_pullback_end": 7}),
    ]
    
    # Profit-taking variants
    profit_configs = [
        ("PT-Default", {"profit_variant": "default"}),
        ("PT-VariantB", {"profit_variant": "variant_b"}),
    ]
    
    # Industry concentration
    industry_configs = [
        ("Top3", {"top_industries": 3}),
        ("Top5", {"top_industries": 5}),
        ("Top7", {"top_industries": 7}),
    ]
    
    base_params = {
        "start_date": "20150101",
        "end_date": "20200101",
        "capital": 1_500_000.0,
        "cost_stress": 1.0,
        "max_positions": 10,
    }
    
    for name, params in configs:
        print(f"\n{name}:")
        p = {**base_params, **params}
        result = BacktestEngine(**p).run()
        all_results.append({"config": name, **result})
    
    # Save sweep results
    out_path = OUTPUT_DIR / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n✅ Sweep complete. Results: {out_path}")
    
    return all_results


def cmd_final(args):
    """Run final holdout evaluation with best parameters."""
    
    print("\n" + "="*70)
    print("FINAL HOLDOUT EVALUATION")
    print("="*70)
    
    # Best parameters (to be determined by sweep + validation)
    # Using default parameters until sweep identifies best
    best_params = {
        "start_date": "20200101",
        "end_date": "20250101",
        "capital": 1_500_000.0,
        "cost_stress": 1.0,
        "top_industries": 5,
        "max_positions": 10,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "n_leg1_start": 20,
        "n_leg1_end": 10,
        "n_pullback_end": 5,
        "profit_variant": "default",
    }
    
    # Normal cost run
    print("\n--- Normal Cost ---")
    engine_normal = BacktestEngine(**best_params)
    results_normal = engine_normal.run()
    
    # 2x cost stress test
    print("\n--- 2× Cost Stress Test ---")
    stress_params = {**best_params, "cost_stress": 2.0}
    engine_stress = BacktestEngine(**stress_params)
    results_stress = engine_stress.run()
    
    # Compare profit variants
    print("\n--- Profit-Taking Variant Comparison ---")
    variant_results = {}
    for variant in ["default", "variant_b"]:
        v_params = {**best_params, "profit_variant": variant}
        engine = BacktestEngine(**v_params)
        variant_results[variant] = engine.run()
        print(f"  {variant}: CAGR={variant_results[variant].get('cagr_pct', 0):.2f}% | "
              f"MaxDD={variant_results[variant].get('max_drawdown_pct', 0):.2f}%")
    
    # Final report
    final_report = {
        "timestamp": datetime.now().isoformat(),
        "holdout_period": "2020-2025",
        "normal_cost": results_normal,
        "stress_cost_2x": results_stress,
        "profit_variants": variant_results,
        "cost_fragility": {
            "cagr_decline_pct": (
                (results_normal.get("cagr_pct", 0) - results_stress.get("cagr_pct", 0))
                / abs(results_normal.get("cagr_pct", 1)) * 100
            ) if results_normal.get("cagr_pct", 0) != 0 else 0,
        },
    }
    
    out_path = OUTPUT_DIR / f"final_holdout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(final_report, indent=2, default=str))
    
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"  Normal Cost CAGR:   {results_normal.get('cagr_pct', 0):.2f}%")
    print(f"  Normal Cost MaxDD:  {results_normal.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Normal Cost Calmar: {results_normal.get('calmar_ratio', 0):.2f}")
    print(f"  Stress Cost CAGR:   {results_stress.get('cagr_pct', 0):.2f}%")
    print(f"  Cost Fragility:     {final_report['cost_fragility']['cagr_decline_pct']:.1f}% CAGR decline")
    print(f"\n  Report: {out_path}")
    print(f"{'='*70}")
    
    return final_report


def cmd_all(args):
    """Run the complete pipeline."""
    print("\n" + "="*70)
    print("COMPLETE RESEARCH PIPELINE")
    print("="*70)
    
    # 1. Build data lake (if needed)
    lake_exists = (Path(__file__).parent / "lake" / "catalog.duckdb").exists()
    if not lake_exists:
        print("\n[1/5] Building data lake...")
        cmd_build()
    else:
        print("\n[1/5] Data lake exists, skipping build.")
    
    # 2. Audit
    print("\n[2/5] Running data audit...")
    cmd_audit()
    
    # 3. Sweep
    print("\n[3/5] Running parameter sweep...")
    cmd_sweep(args)
    
    # 4. Validate
    print("\n[4/5] Running validation suite...")
    cmd_validate(args)
    
    # 5. Final holdout
    print("\n[5/5] Running final holdout...")
    cmd_final(args)
    
    print("\n✅ Complete pipeline finished.")


def main():
    parser = argparse.ArgumentParser(
        description="A-Share Trend Following Strategy Research Pipeline"
    )
    parser.add_argument(
        "command",
        choices=["build", "audit", "backtest", "validate", "sweep", "final", "all"],
        help="Pipeline stage to run"
    )
    parser.add_argument("--start", help="Start date (YYYYMMDD)")
    parser.add_argument("--end", help="End date (YYYYMMDD)")
    parser.add_argument("--stress", type=float, help="Cost stress multiplier")
    parser.add_argument("--industries", type=int, help="Top N industries")
    parser.add_argument("--positions", type=int, help="Max positions")
    parser.add_argument("--profit-variant", choices=["default", "variant_b"],
                       help="Profit-taking variant")
    
    args = parser.parse_args()
    
    commands = {
        "build": cmd_build,
        "audit": cmd_audit,
        "backtest": lambda: cmd_backtest(args),
        "validate": lambda: cmd_validate(args),
        "sweep": lambda: cmd_sweep(args),
        "final": lambda: cmd_final(args),
        "all": lambda: cmd_all(args),
    }
    
    commands[args.command]()


if __name__ == "__main__":
    main()
