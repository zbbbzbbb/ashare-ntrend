"""
Final comparison of improvement candidates, in-sample (2015-2025) AND on the
untouched out-of-sample window (2025-01-01 .. 2026-06-24).

Candidates:
  ORIG          original baseline (bbi2 + pt20, full-bullish regime)
  TIME_STOP_h   keep the funnel but exit after h days (harvest the 10d pop)
  ETF_TIMING    long CSI300 when MACD full-bullish, else cash (drop stock picking)

Also: 2x cost stress + realistic full-cost (slippage+impact) on the winner.
"""
import warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); np.seterr(all="ignore")

import research.fast_bt as fb
from research.fast_bt import Config, run_backtest, regime_series

WIN = {"IS 2015-2025": ("20150101", "20250101"),
       "OOS 2025-2026": ("20250101", "20260624")}


def etf_timing(start, end, full_bullish=True, cost=0.0006):
    s = next(i for i, d in enumerate(fb.DATES) if d >= start)
    e = next((i for i, d in enumerate(fb.DATES) if d >= end), len(fb.DATES) - 1)
    c = pd.Series(fb.CSI_CLOSE); ret = c.pct_change().fillna(0).to_numpy()
    sig = np.roll(regime_series(full_bullish=full_bullish).astype(float), 1); sig[0] = 0
    switches = np.abs(np.diff(sig, prepend=sig[0]))
    nav = np.cumprod((1 + sig * ret - switches * cost)[s:e + 1])
    years = len(nav) / 252; tot = nav[-1] / nav[0] - 1
    cagr = (1 + tot) ** (1 / years) - 1
    peak = np.maximum.accumulate(nav); dd = (nav - peak) / peak
    rets = np.diff(nav) / nav[:-1]
    return {"name": "ETF_TIMING (CSI300/cash)", "cagr_pct": round(100 * cagr, 2),
            "max_dd_pct": round(100 * dd.min(), 2),
            "calmar": round(cagr / abs(dd.min()), 2) if dd.min() else 0,
            "sharpe": round(rets.mean()/rets.std()*np.sqrt(252), 2) if rets.std() else 0,
            "total_ret_pct": round(100 * tot, 2), "n_trades": int(switches[s:e+1].sum()//2),
            "win_rate_pct": "-", "avg_hold": "-", "best_pct": "-",
            "final_nav": round(1_500_000 * (1 + tot))}


def row(r):
    return (f"{r['name']:<30}{r['cagr_pct']:>7}{r['max_dd_pct']:>8}{str(r['calmar']):>7}"
            f"{str(r['sharpe']):>6}{str(r.get('profit_factor','-')):>6}{str(r['win_rate_pct']):>6}"
            f"{str(r['n_trades']):>5}{str(r['avg_hold']):>6}{str(r['best_pct']):>7}{r['final_nav']:>12,}")


def main():
    fb.build_panels()
    fb.compute_stock_signals(12, 26, 9, verbose=False)
    fb.compute_industry_scores(12, 26, 9, verbose=False)

    out = {}
    for label, (s, e) in WIN.items():
        cfgs = [
            Config(name="ORIG baseline (bbi2+pt)", start=s, end=e, exit_rule="bbi2", profit_take=True),
            Config(name="TIME_STOP 5d", start=s, end=e, exit_rule="time_stop", hold_days=5, profit_take=False),
            Config(name="TIME_STOP 8d", start=s, end=e, exit_rule="time_stop", hold_days=8, profit_take=False),
            Config(name="TIME_STOP 10d", start=s, end=e, exit_rule="time_stop", hold_days=10, profit_take=False),
            Config(name="TIME_STOP 15d", start=s, end=e, exit_rule="time_stop", hold_days=15, profit_take=False),
            Config(name="TIME_STOP 10d +realcost", start=s, end=e, exit_rule="time_stop", hold_days=10,
                   profit_take=False, slippage=0.001, impact=0.0005),
            Config(name="TIME_STOP 10d 2x-cost", start=s, end=e, exit_rule="time_stop", hold_days=10,
                   profit_take=False, slippage=0.001, impact=0.0005, cost_stress=2.0),
        ]
        res = [run_backtest(c) for c in cfgs]
        res.append(etf_timing(s, e))
        out[label] = res

    hdr = (f"{'candidate':<30}{'CAGR':>7}{'MaxDD':>8}{'Calmar':>7}{'Shrp':>6}{'PF':>6}"
           f"{'Win%':>6}{'Trd':>5}{'Hold':>6}{'Best%':>7}{'Final':>12}")
    for label, res in out.items():
        print(f"\n{'='*len(hdr)}\n{label}\n{'='*len(hdr)}")
        print(hdr); print("-" * len(hdr))
        for r in res:
            print(row(r))
    from pathlib import Path
    p = Path(__file__).parent.parent / "output" / "final_compare.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
