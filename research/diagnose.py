"""
Diagnostics to locate where (if anywhere) edge lives:
  1. Entry event-study: forward returns after the FULL entry signal fires,
     vs a matched baseline (all valid stocks on the same bullish-regime days).
  2. Index-timing benchmark: does the CSI300 MACD-regime timing layer add value
     when expressed simply as a long CSI300 / cash overlay?
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore"); np.seterr(all="ignore")

import research.fast_bt as fb
from research.fast_bt import regime_series


def build_top_industry_mask(top_n=5):
    """[T,N] bool: stock j is a constituent of a top-N industry on day di."""
    T = len(fb.DATES); N = len(fb.COLS)
    mask = np.zeros((T, N), dtype=bool)
    order_all = np.argsort(-np.nan_to_num(fb.IND_SCORE, nan=-1e9), axis=1)
    for di in range(T):
        mb = fb.IND_MACD_BULL[di]
        order = order_all[di]
        top = [m for m in order if mb[m]][:top_n]
        if len(top) < top_n:
            for m in order:
                if m not in top:
                    top.append(m)
                    if len(top) >= top_n: break
        for m in top[:top_n]:
            mask[di, fb.IND_COLS[fb.IND_CODE_ORDER[m]]] = True
    return mask


def event_study(start="20150101", end="20250101", horizons=(5, 10, 20, 40)):
    s = next(i for i, d in enumerate(fb.DATES) if d >= start)
    e = next((i for i, d in enumerate(fb.DATES) if d >= end), len(fb.DATES) - 1)
    regime = regime_series(full_bullish=True)
    top = build_top_industry_mask(5)
    C = fb.CLOSE

    # full entry signal cells
    entry = (regime[:, None] & top & fb.S_VALID & fb.S_MACD_BULL & fb.S_VOLOK & fb.S_ISN)
    # baseline: any valid stock on a bullish-regime day (the opportunity set)
    base = (regime[:, None] & fb.S_VALID)

    rows = []
    for k in horizons:
        fwd = np.full_like(C, np.nan)
        fwd[:-k] = C[k:] / C[:-k] - 1.0
        win = slice(s, e - k)
        em = entry[win] & ~np.isnan(fwd[win])
        bm = base[win] & ~np.isnan(fwd[win])
        ev = fwd[win][em]; bv = fwd[win][bm]
        rows.append({
            "horizon_d": k,
            "n_entry": int(em.sum()),
            "entry_mean_%": round(100 * np.mean(ev), 2) if ev.size else None,
            "entry_median_%": round(100 * np.median(ev), 2) if ev.size else None,
            "entry_winrate_%": round(100 * np.mean(ev > 0), 1) if ev.size else None,
            "base_mean_%": round(100 * np.mean(bv), 2) if bv.size else None,
            "base_winrate_%": round(100 * np.mean(bv > 0), 1) if bv.size else None,
            "edge_mean_bps": round(1e4 * (np.mean(ev) - np.mean(bv)), 1) if ev.size and bv.size else None,
        })
    return pd.DataFrame(rows)


def _perf(nav, dates):
    nav = np.asarray(nav, float)
    years = len(nav) / 252
    tot = nav[-1] / nav[0] - 1
    cagr = (1 + tot) ** (1 / years) - 1
    peak = np.maximum.accumulate(nav); dd = (nav - peak) / peak
    rets = np.diff(nav) / nav[:-1]
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    return {"cagr_%": round(100 * cagr, 2), "maxdd_%": round(100 * dd.min(), 2),
            "calmar": round(cagr / abs(dd.min()), 2) if dd.min() != 0 else 0,
            "sharpe": round(sharpe, 2), "total_%": round(100 * tot, 2)}


def index_timing(start="20150101", end="20260624"):
    s = next(i for i, d in enumerate(fb.DATES) if d >= start)
    e = next((i for i, d in enumerate(fb.DATES) if d >= end), len(fb.DATES) - 1)
    c = pd.Series(fb.CSI_CLOSE)
    ret = c.pct_change().fillna(0).to_numpy()
    cost = 0.0006  # round-trip-ish ETF cost applied on regime switches

    def overlay(signal):
        # position is decided at close of day t-1, earns day t's return (no look-ahead)
        sig = np.roll(signal.astype(float), 1); sig[0] = 0
        switches = np.abs(np.diff(sig, prepend=sig[0]))
        daily = 1 + sig * ret - switches * cost
        return np.cumprod(daily[s:e + 1])

    bh = np.cumprod((1 + ret)[s:e + 1])
    ma200 = (c > c.rolling(200).mean()).to_numpy()
    gc = (c.rolling(50).mean() > c.rolling(200).mean()).to_numpy()
    full_bull = regime_series(full_bullish=True)
    relaxed = regime_series(full_bullish=False)

    out = {
        "CSI300 buy&hold": _perf(bh, None),
        "CSI300 > MA200": _perf(overlay(ma200), None),
        "Golden cross 50/200": _perf(overlay(gc), None),
        "MACD full-bullish (orig timing)": _perf(overlay(full_bull), None),
        "MACD relaxed DIF>DEA": _perf(overlay(relaxed), None),
    }
    pct = lambda sig: round(100 * sig[s:e + 1].mean(), 1)
    expo = {"CSI300 > MA200": pct(ma200), "Golden cross 50/200": pct(gc),
            "MACD full-bullish (orig timing)": pct(full_bull), "MACD relaxed DIF>DEA": pct(relaxed)}
    return out, expo


def main():
    fb.build_panels()
    fb.compute_stock_signals(12, 26, 9, verbose=False)
    fb.compute_industry_scores(12, 26, 9, verbose=False)

    print("\n" + "=" * 78)
    print("ENTRY EVENT-STUDY  (forward return after full entry signal vs baseline)")
    print("In-sample 2015-2025, bullish-regime days. 'edge' = entry_mean - base_mean")
    print("=" * 78)
    print(event_study().to_string(index=False))

    print("\n" + "=" * 78)
    print("INDEX-TIMING BENCHMARK  (CSI300, 2015-01 .. 2026-06, net of switch cost)")
    print("=" * 78)
    perf, expo = index_timing()
    print(f"{'strategy':<34}{'CAGR%':>7}{'MaxDD%':>8}{'Calmar':>7}{'Sharpe':>7}{'Total%':>8}{'Expo%':>7}")
    for k, v in perf.items():
        print(f"{k:<34}{v['cagr_%']:>7}{v['maxdd_%']:>8}{v['calmar']:>7}{v['sharpe']:>7}{v['total_%']:>8}{expo.get(k,100):>7}")


if __name__ == "__main__":
    main()
