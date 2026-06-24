"""
Volatility / Sharpe / Sortino for the key candidates, IS and OOS.

Notes on convention:
  * daily returns are of the NAV (cash periods earn 0% here -> CONSERVATIVE;
    in reality idle cash earns ~2% money-market, which would lift CAGR & Sharpe)
  * ann_vol = std(daily) * sqrt(252)
  * Sharpe = (ann_arith_return - rf) / ann_vol,  reported at rf=0 and rf=2.5%
  * Sortino uses downside deviation (negative daily returns only)
"""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); np.seterr(all="ignore")

from research.explore import load_indices, signals_for, INDICES
import research.fast_bt as fb
from research.fast_bt import Config, run_backtest, regime_series

COST = 0.0006
RF = 0.025


def stats(nav):
    nav = np.asarray(nav, float)
    yrs = len(nav) / 252
    tot = nav[-1] / nav[0] - 1
    cagr = (1 + tot) ** (1 / yrs) - 1
    r = np.diff(nav) / nav[:-1]
    ann_ret = r.mean() * 252                      # arithmetic annualized
    ann_vol = r.std() * np.sqrt(252)
    downside = r[r < 0].std() * np.sqrt(252) if (r < 0).any() else np.nan
    peak = np.maximum.accumulate(nav); dd = (nav - peak) / peak
    return dict(
        cagr=round(100 * cagr, 2), ann_vol=round(100 * ann_vol, 1),
        sharpe0=round(ann_ret / ann_vol, 2) if ann_vol else 0,
        sharpe_rf=round((ann_ret - RF) / ann_vol, 2) if ann_vol else 0,
        sortino=round((ann_ret - RF) / downside, 2) if downside and not np.isnan(downside) else 0,
        maxdd=round(100 * dd.min(), 2),
        calmar=round(cagr / abs(dd.min()), 2) if dd.min() else 0,
    )


def overlay_nav(close_col, sig, s, e):
    ret = np.nan_to_num(pd.Series(close_col).pct_change().fillna(0).to_numpy())
    pos = np.roll(sig.astype(float), 1); pos[0] = 0
    sw = np.abs(np.diff(pos, prepend=pos[0]))
    return np.cumprod((1 + pos * ret - sw * COST)[s:e + 1]), round(100 * pos[s:e + 1].mean(), 1)


def main():
    dates, didx, codes, close = load_indices()
    T = len(dates); cmap = {INDICES[c]: k for k, c in enumerate(codes)}
    si = next(i for i, d in enumerate(dates) if d >= "20150101")
    ei = next(i for i, d in enumerate(dates) if d >= "20250101")
    oi = ei
    oe = next((i for i, d in enumerate(dates) if d >= "20260624"), T - 1)

    # curated timing candidates (index_label, signal_label or None=buy&hold)
    cand = [
        ("中证1000", "MACD三条件(12,26)"),
        ("沪深300", "MACD三条件(12,26)"),
        ("中证500", "MACD看多DIF>DEA(12,26)"),
        ("创业板指", "close>MA50"),
        ("沪深300", None),
        ("中证1000", None),
    ]
    rows = []
    for idx_lbl, sig_lbl in cand:
        k = cmap[idx_lbl]
        sig = np.ones(T, bool) if sig_lbl is None else signals_for(close[:, k])[sig_lbl]
        nav_is, ex_is = overlay_nav(close[:, k], sig, si, ei)
        nav_oo, ex_oo = overlay_nav(close[:, k], sig, oi, oe)
        name = f"{idx_lbl} · {'买入持有' if sig_lbl is None else sig_lbl}"
        rows.append((name, ex_is, stats(nav_is), ex_oo, stats(nav_oo)))

    hdr =(f"{'候选策略':<30}{'暴露%':>6}{'CAGR%':>7}{'年化波动%':>9}{'Sharpe(0)':>10}{'Sharpe(2.5%)':>13}{'Sortino':>8}{'回撤%':>8}{'Calmar':>7}")
    print("\n" + "=" * len(hdr)); print("样本内 IS 2015-2025"); print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for name, ex, s_is, _, _ in rows:
        print(f"{name:<30}{ex:>6}{s_is['cagr']:>7}{s_is['ann_vol']:>9}{s_is['sharpe0']:>10}{s_is['sharpe_rf']:>13}{s_is['sortino']:>8}{s_is['maxdd']:>8}{s_is['calmar']:>7}")
    print("\n" + "=" * len(hdr)); print("样本外 OOS 2025-2026"); print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for name, _, _, ex2, s_oo in rows:
        print(f"{name:<30}{ex2:>6}{s_oo['cagr']:>7}{s_oo['ann_vol']:>9}{s_oo['sharpe0']:>10}{s_oo['sharpe_rf']:>13}{s_oo['sortino']:>8}{s_oo['maxdd']:>8}{s_oo['calmar']:>7}")


if __name__ == "__main__":
    main()
