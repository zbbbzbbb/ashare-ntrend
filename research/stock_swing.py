"""
Can a STOCK strategy work if we respect what the data says?
  - edge is a 5-10 day momentum pop -> trade short swings, fast exits
  - cost ~0.4% round trip eats small edges -> be selective: fewer, bigger-move,
    higher-conviction entries (stronger volume, own-uptrend filter, real N only)
All variants: CSI300 full-bullish gate, REALISTIC full costs (slip+impact), IS + OOS.
"""
import warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); np.seterr(all="ignore")

import research.fast_bt as fb
from research.fast_bt import Config, run_backtest, regime_series

REAL = dict(slippage=0.001, impact=0.0005)   # realistic full cost on top of comm+stamp
WIN = {"IS 2015-2025": ("20150101", "20250101"), "OOS 2025-2026": ("20250101", "20260624")}


def etf_timing(start, end, cost=0.0006):
    s = next(i for i, d in enumerate(fb.DATES) if d >= start)
    e = next((i for i, d in enumerate(fb.DATES) if d >= end), len(fb.DATES) - 1)
    c = pd.Series(fb.CSI_CLOSE); ret = c.pct_change().fillna(0).to_numpy()
    sig = np.roll(regime_series(True).astype(float), 1); sig[0] = 0
    sw = np.abs(np.diff(sig, prepend=sig[0]))
    nav = np.cumprod((1 + sig * ret - sw * cost)[s:e + 1])
    yrs = len(nav) / 252; tot = nav[-1] / nav[0] - 1; cagr = (1 + tot) ** (1 / yrs) - 1
    peak = np.maximum.accumulate(nav); dd = (nav - peak) / peak
    return {"name": "[参照] ETF择时", "cagr_pct": round(100*cagr,2), "max_dd_pct": round(100*dd.min(),2),
            "calmar": round(cagr/abs(dd.min()),2), "profit_factor": "-", "win_rate_pct": "-",
            "n_trades": int(sw[s:e+1].sum()//2), "avg_hold": "-", "best_pct": "-",
            "final_nav": round(1_500_000*(1+tot))}


def row(r):
    return (f"{r['name']:<34}{r['cagr_pct']:>7}{r['max_dd_pct']:>8}{str(r['calmar']):>7}"
            f"{str(r.get('profit_factor','-')):>6}{str(r['win_rate_pct']):>6}{str(r['n_trades']):>5}"
            f"{str(r['avg_hold']):>6}{str(r['best_pct']):>7}{r['final_nav']:>12,}")


def main():
    fb.build_panels()
    fb.compute_stock_signals(12, 26, 9, verbose=False)
    fb.compute_industry_scores(12, 26, 9, verbose=False)

    out = {}
    for label, (s, e) in WIN.items():
        C = lambda **k: Config(start=s, end=e, **{**REAL, **k})
        cfgs = [
            C(name="ORIG (BBI止损,真实成本)", exit_rule="bbi2", profit_take=True, slippage=0.001, impact=0.0005),
            C(name="A 纯5天时间止损", exit_rule="time_stop", hold_days=5, profit_take=False),
            C(name="B 摆动:+8%/−5%/5天", exit_rule="swing", take_profit=0.08, hard_stop=-0.05, hold_days=5, profit_take=False),
            C(name="C 摆动:+10%/−6%/8天", exit_rule="swing", take_profit=0.10, hard_stop=-0.06, hold_days=8, profit_take=False),
            # selective: real N only + strong volume + own-uptrend filter
            C(name="D 严选(真N+量1.5+MA60)5天", exit_rule="time_stop", hold_days=5, profit_take=False,
              relax_n=False, vol_mult=1.5, entry_trend_ma=60),
            C(name="E 严选+摆动:+8%/−5%/5天", exit_rule="swing", take_profit=0.08, hard_stop=-0.05, hold_days=5,
              profit_take=False, relax_n=False, vol_mult=1.5, entry_trend_ma=60),
            C(name="F 极严选(真N+量2.0+MA60)+摆动", exit_rule="swing", take_profit=0.08, hard_stop=-0.05, hold_days=5,
              profit_take=False, relax_n=False, vol_mult=2.0, entry_trend_ma=60),
            C(name="G 严选+多买(4笔/周)摆动", exit_rule="swing", take_profit=0.08, hard_stop=-0.05, hold_days=5,
              profit_take=False, relax_n=False, vol_mult=1.5, entry_trend_ma=60, max_weekly_buys=4),
        ]
        res = [run_backtest(c) for c in cfgs]
        res.insert(0, etf_timing(s, e))
        out[label] = res

    hdr = (f"{'方案':<34}{'CAGR':>7}{'回撤':>8}{'Calmar':>7}{'PF':>6}{'胜率':>6}{'笔数':>5}{'持仓':>6}{'最大盈':>7}{'终值':>12}")
    for label, res in out.items():
        print(f"\n{'='*len(hdr)}\n{label}（均为真实全成本：佣金+印花+滑点0.1%+冲击0.05%）\n{'='*len(hdr)}")
        print(hdr); print("-" * len(hdr))
        for r in res:
            print(row(r))
    from pathlib import Path
    p = Path(__file__).parent.parent / "output" / "stock_swing.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
