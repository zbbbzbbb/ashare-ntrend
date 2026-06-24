"""
Improvement sweep. Builds panels once, then fans configs across cores via fork
(panels/signals shared copy-on-write). Configs are grouped by MACD params so the
heavy signal panels are recomputed only once per group.
"""
import warnings, json, multiprocessing as mp
from collections import defaultdict
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

import research.fast_bt as fb
from research.fast_bt import Config, run_backtest

OUT = Path(__file__).parent.parent / "output"


def make_configs(start="20150101", end="20250101"):
    C = lambda **k: Config(start=start, end=end, **k)
    cfgs = []
    # ---- Stage 1: isolate EXIT rule (regime=orig full-bullish, util=orig 2/wk) ----
    cfgs += [
        C(name="00_baseline (bbi2+pt20)", exit_rule="bbi2", profit_take=True),
        C(name="01_bbi2 no-profit-take", exit_rule="bbi2", profit_take=False),
        C(name="02_chandelier k3", exit_rule="chandelier", chandelier_k=3.0, profit_take=False),
        C(name="03_chandelier k4", exit_rule="chandelier", chandelier_k=4.0, profit_take=False),
        C(name="04_chandelier k3 +pt", exit_rule="chandelier", chandelier_k=3.0, profit_take=True),
        C(name="05_MA20 trail", exit_rule="ma_trail", ma_trail=20, profit_take=False),
        C(name="06_MA50 trail", exit_rule="ma_trail", ma_trail=50, profit_take=False),
    ]
    # ---- Stage 2: capital UTILIZATION (relax regime / more buys/week) ----
    cfgs += [
        C(name="10_relaxed-regime bbi2+pt", exit_rule="bbi2", profit_take=True, full_bullish=False),
        C(name="11_relaxed-regime chand3", exit_rule="chandelier", chandelier_k=3.0, profit_take=False, full_bullish=False),
        C(name="12_chand3 4buys/wk", exit_rule="chandelier", chandelier_k=3.0, profit_take=False, max_weekly_buys=4),
        C(name="13_relaxed chand3 4buys/wk", exit_rule="chandelier", chandelier_k=3.0, profit_take=False, full_bullish=False, max_weekly_buys=4),
        C(name="14_relaxed chand3 4buys/wk 6buys", exit_rule="chandelier", chandelier_k=3.0, profit_take=False, full_bullish=False, max_weekly_buys=6),
        C(name="15_relaxed MA50 4buys/wk", exit_rule="ma_trail", ma_trail=50, profit_take=False, full_bullish=False, max_weekly_buys=4),
        C(name="16_relaxed chand4 4buys/wk", exit_rule="chandelier", chandelier_k=4.0, profit_take=False, full_bullish=False, max_weekly_buys=4),
    ]
    # ---- Stage 3: MACD(20,38) group (report's headline params) ----
    cfgs += [
        C(name="20_M20/38 baseline (report)", exit_rule="bbi2", profit_take=True, macd_fast=20, macd_slow=38),
        C(name="21_M20/38 chand3", exit_rule="chandelier", chandelier_k=3.0, profit_take=False, macd_fast=20, macd_slow=38),
        C(name="22_M20/38 relaxed chand3 4buys", exit_rule="chandelier", chandelier_k=3.0, profit_take=False, full_bullish=False, max_weekly_buys=4, macd_fast=20, macd_slow=38),
    ]
    return cfgs


def run_group(params, cfgs):
    fb.compute_stock_signals(*params, verbose=False)
    fb.compute_industry_scores(*params, verbose=False)
    # fork pool inherits the freshly-built global panels copy-on-write
    with mp.get_context("fork").Pool(processes=max(1, mp.cpu_count() - 2)) as pool:
        return pool.map(run_backtest, cfgs)


def main():
    fb.build_panels()
    cfgs = make_configs()
    groups = defaultdict(list)
    for c in cfgs:
        groups[(c.macd_fast, c.macd_slow, c.macd_signal)].append(c)

    results = []
    for params, cs in groups.items():
        print(f"\n>>> MACD group {params}: {len(cs)} configs")
        results += run_group(params, cs)

    results.sort(key=lambda r: r["name"])
    cols = ["name", "cagr_pct", "max_dd_pct", "calmar", "sharpe", "profit_factor",
            "win_rate_pct", "n_trades", "avg_hold", "avg_win_hold", "best_pct", "final_nav"]
    hdr = f"{'config':<32}{'CAGR':>7}{'MaxDD':>8}{'Calmar':>7}{'Shrp':>6}{'PF':>6}{'Win%':>6}{'Trd':>5}{'Hold':>6}{'WHold':>6}{'Best%':>7}{'Final':>11}"
    print("\n" + "=" * len(hdr)); print(hdr); print("=" * len(hdr))
    for r in results:
        print(f"{r['name']:<32}{r['cagr_pct']:>7}{r['max_dd_pct']:>8}{r['calmar']:>7}"
              f"{r['sharpe']:>6}{r['profit_factor']:>6}{r['win_rate_pct']:>6}{r['n_trades']:>5}"
              f"{r['avg_hold']:>6}{r['avg_win_hold']:>6}{r['best_pct']:>7}{r['final_nav']:>11,}")
    (OUT / "improve_sweep.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {OUT/'improve_sweep.json'}")


if __name__ == "__main__":
    main()
