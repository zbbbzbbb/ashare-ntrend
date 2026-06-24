"""
Open up the design space (keep the IDEA, drop the specific knobs).

PART 1 — HOW TO RATE THE MARKET: sweep indices x timing signals
        (MA periods, time-series momentum, MACD variants, regime+trend combos),
        expressed as long-index / cash overlay. Timing is where the value is.
PART 2 — HOW TO RANK STOCKS: forward-return event study comparing the N-pattern
        against momentum / 52w-high breakout / trend signals. Which has real edge?
"""
import warnings
import numpy as np, pandas as pd, duckdb
from pathlib import Path
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
CATALOG = str(Path.home() / "quant-data" / "catalog.duckdb")

INDICES = {
    "000300.SH": "沪深300", "000905.SH": "中证500", "000852.SH": "中证1000",
    "399006.SZ": "创业板指", "000001.SH": "上证综指", "000985.CSI": "中证全指",
}
COST = 0.0006


# ---------------- shared calendar + index loader ----------------
def load_indices():
    con = duckdb.connect(CATALOG, read_only=True)
    cal = con.execute("SELECT cal_date FROM trade_cal WHERE is_open=1 AND cal_date>=DATE '2013-01-01' AND cal_date<=DATE '2026-06-24' ORDER BY cal_date").fetchdf()
    dates = [str(d)[:10].replace("-", "") for d in cal["cal_date"]]
    didx = {d: i for i, d in enumerate(dates)}
    T = len(dates); codes = list(INDICES)
    close = np.full((T, len(codes)), np.nan)
    for k, c in enumerate(codes):
        df = con.execute("SELECT trade_date, close FROM index_daily WHERE ts_code=? AND trade_date>=DATE '2013-01-01' ORDER BY trade_date", [c]).fetchdf()
        for d, v in zip(df["trade_date"], df["close"]):
            i = didx.get(str(d)[:10].replace("-", ""))
            if i is not None: close[i, k] = v
    con.close()
    return dates, didx, codes, pd.DataFrame(close).ffill().to_numpy()


def signals_for(close_col):
    """dict of timing-signal name -> boolean array (long when True)."""
    c = pd.Series(close_col)
    out = {}
    for n in (50, 100, 150, 200):
        out[f"close>MA{n}"] = (c > c.rolling(n).mean()).to_numpy()
    for n, lbl in ((60, "3m"), (120, "6m"), (250, "12m")):
        out[f"mom>0 {lbl}"] = (c / c.shift(n) - 1 > 0).to_numpy()
    out["MA50>MA200(金叉)"] = (c.rolling(50).mean() > c.rolling(200).mean()).to_numpy()
    # MACD variants
    for f, s in ((12, 26), (20, 38)):
        dif = c.ewm(span=f, adjust=False).mean() - c.ewm(span=s, adjust=False).mean()
        dea = dif.ewm(span=9, adjust=False).mean(); hist = (dif - dea) * 2
        out[f"MACD看多DIF>DEA({f},{s})"] = (dif > dea).to_numpy()
        out[f"MACD三条件({f},{s})"] = ((dif > dea) & (hist.abs() > hist.abs().shift(1)) & (dif > dif.shift(1))).to_numpy()
    # combos: regime AND trend
    ma200 = (c > c.rolling(200).mean()).to_numpy()
    dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    out["MACD看多 且 >MA200"] = ((dif > dif.ewm(span=9, adjust=False).mean()).to_numpy() & ma200)
    out["mom6m>0 且 >MA200"] = ((c / c.shift(120) - 1 > 0).to_numpy() & ma200)
    return out


def overlay(close_col, sig, s, e):
    ret = np.nan_to_num(pd.Series(close_col).pct_change().fillna(0).to_numpy())
    pos = np.roll(sig.astype(float), 1); pos[0] = 0
    sw = np.abs(np.diff(pos, prepend=pos[0]))
    nav = np.cumprod((1 + pos * ret - sw * COST)[s:e + 1])
    yrs = len(nav) / 252; tot = nav[-1] / nav[0] - 1; cagr = (1 + tot) ** (1 / yrs) - 1
    peak = np.maximum.accumulate(nav); dd = (nav - peak) / peak
    r = np.diff(nav) / nav[:-1]
    return dict(cagr=round(100 * cagr, 2), dd=round(100 * dd.min(), 2),
                calmar=round(cagr / abs(dd.min()), 2) if dd.min() else 0,
                sharpe=round(r.mean() / r.std() * np.sqrt(252), 2) if r.std() else 0,
                expo=round(100 * pos[s:e + 1].mean(), 1))


def part1():
    dates, didx, codes, close = load_indices()
    T = len(dates)
    si = next(i for i, d in enumerate(dates) if d >= "20150101")
    ei = next(i for i, d in enumerate(dates) if d >= "20250101")
    oi = next(i for i, d in enumerate(dates) if d >= "20250101")
    oe = next((i for i, d in enumerate(dates) if d >= "20260624"), T - 1)

    rows = []
    for k, code in enumerate(codes):
        if np.isnan(close[si, k]):  # index has no data at IS start (e.g. CSI1000 ok, STAR no)
            pass
        bh = overlay(close[:, k], np.ones(T, bool), si, ei)
        rows.append(("买入持有", INDICES[code], bh, overlay(close[:, k], np.ones(T, bool), oi, oe)))
        for name, sig in signals_for(close[:, k]).items():
            rows.append((name, INDICES[code], overlay(close[:, k], sig, si, ei),
                         overlay(close[:, k], sig, oi, oe)))

    # rank by IS Calmar, show top 18
    rows.sort(key=lambda r: -r[2]["calmar"])
    print("\n" + "=" * 104)
    print("PART 1 · 择时扫描：指数 × 信号  (按样本内Calmar排序，Top 20；含样本外验证)")
    print("=" * 104)
    h = f"{'信号':<22}{'指数':<9}{'IS_CAGR':>8}{'IS_DD':>8}{'IS_Cal':>7}{'IS_Shrp':>8}{'暴露%':>7}  | {'OOS_CAGR':>9}{'OOS_DD':>8}{'OOS_Cal':>8}"
    print(h); print("-" * len(h))
    for name, idx, is_, oos in rows[:20]:
        print(f"{name:<22}{idx:<9}{is_['cagr']:>8}{is_['dd']:>8}{is_['calmar']:>7}{is_['sharpe']:>8}{is_['expo']:>7}  | "
              f"{oos['cagr']:>9}{oos['dd']:>8}{oos['calmar']:>8}")


# ---------------- PART 2: stock signal forward-edge ----------------
def part2():
    import research.fast_bt as fb
    if fb.CLOSE is None:
        fb.build_panels(verbose=False)
        fb.compute_stock_signals(12, 26, 9, verbose=False)
    C = fb.CLOSE
    c = pd.DataFrame(C)
    regime = fb.regime_series(full_bullish=True)
    valid = fb.S_VALID

    ma50 = c.rolling(50).mean().to_numpy(); ma200 = c.rolling(200).mean().to_numpy()
    mom120 = (C / np.roll(C, 120, axis=0) - 1); mom120[:120] = np.nan
    hi250 = pd.DataFrame(C).rolling(250).max().shift(1).to_numpy()
    near_hi = C >= hi250 * 0.98          # within 2% of 1y high
    new_hi = C > hi250                   # fresh 1y breakout
    # cross-sectional momentum top-quintile each day
    mom_rank = pd.DataFrame(mom120).rank(axis=1, pct=True).to_numpy()

    sigs = {
        "N型(原信号)": valid & fb.S_MACD_BULL & fb.S_VOLOK & fb.S_ISN,
        "6月动量Top20%": valid & (mom_rank >= 0.8),
        "近1年新高(2%内)": valid & near_hi,
        "突破1年新高+放量": valid & new_hi & fb.S_VOLOK,
        "趋势(>MA50>MA200)": valid & (C > ma50) & (ma50 > ma200),
        "动量Top20%+>MA200": valid & (mom_rank >= 0.8) & (C > ma200),
    }
    s = next(i for i, d in enumerate(fb.DATES) if d >= "20150101")
    e = next(i for i, d in enumerate(fb.DATES) if d >= "20250101")
    base = (regime[:, None] & valid)

    print("\n" + "=" * 92)
    print("PART 2 · 选股信号前瞻收益对比 (牛市日, 样本内2015-2025; edge=信号-基准, 20日)")
    print("=" * 92)
    hh = f"{'信号':<22}{'笔数':>8}{'10日均值%':>10}{'10日中位%':>10}{'20日均值%':>10}{'20日胜率%':>10}{'20日超额bp':>11}"
    print(hh); print("-" * len(hh))
    for k in (10, 20):
        pass
    for name, mask in sigs.items():
        full = regime[:, None] & mask
        f10 = np.full_like(C, np.nan); f10[:-10] = C[10:] / C[:-10] - 1
        f20 = np.full_like(C, np.nan); f20[:-20] = C[20:] / C[:-20] - 1
        w = slice(s, e - 20)
        m10 = full[w] & ~np.isnan(f10[w]); m20 = full[w] & ~np.isnan(f20[w])
        bm20 = base[w] & ~np.isnan(f20[w])
        v10 = f10[w][m10]; v20 = f20[w][m20]; b20 = f20[w][bm20]
        if v20.size == 0:
            continue
        print(f"{name:<22}{m20.sum():>8}{100*np.mean(v10):>10.2f}{100*np.median(v10):>10.2f}"
              f"{100*np.mean(v20):>10.2f}{100*np.mean(v20>0):>10.1f}{1e4*(np.mean(v20)-np.mean(b20)):>11.1f}")


if __name__ == "__main__":
    part1()
    part2()
