"""
Does broadening from a single CSI300 ETF to a basket of broad-index ETFs help?

Same MACD full-bullish timing, but applied to several real broad indices, then
rotate/equal-weight into the qualifying ones. Real index data (these ARE in the
lake, unlike SW industry indices). T+1, switch cost on turnover.
"""
import warnings
import numpy as np, pandas as pd, duckdb
from pathlib import Path
warnings.filterwarnings("ignore"); np.seterr(all="ignore")

CATALOG = str(Path.home() / "quant-data" / "catalog.duckdb")
BASKET = {
    "000300.SH": "沪深300", "000905.SH": "中证500", "000852.SH": "中证1000",
    "399006.SZ": "创业板指", "000016.SH": "上证50", "000922.CSI": "中证红利",
}
COST = 0.0006  # per unit turnover on rebalance


def load():
    con = duckdb.connect(CATALOG, read_only=True)
    cal = con.execute("SELECT cal_date FROM trade_cal WHERE is_open=1 AND cal_date>=DATE '2014-01-01' AND cal_date<=DATE '2026-06-24' ORDER BY cal_date").fetchdf()
    dates = [str(d)[:10].replace("-", "") for d in cal["cal_date"]]
    didx = {d: i for i, d in enumerate(dates)}
    T = len(dates); codes = list(BASKET)
    close = np.full((T, len(codes)), np.nan)
    for k, c in enumerate(codes):
        df = con.execute("SELECT trade_date, close FROM index_daily WHERE ts_code=? AND trade_date>=DATE '2014-01-01' ORDER BY trade_date", [c]).fetchdf()
        for d, v in zip(df["trade_date"], df["close"]):
            i = didx.get(str(d)[:10].replace("-", ""))
            if i is not None: close[i, k] = v
    con.close()
    return dates, didx, codes, pd.DataFrame(close).ffill().to_numpy()


def macd_bull(close_col, fast=12, slow=26, sig=9):
    c = pd.Series(close_col)
    dif = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    dea = dif.ewm(span=sig, adjust=False).mean()
    hist = (dif - dea) * 2
    return ((dif > dea) & (hist.abs() > hist.abs().shift(1)) & (dif > dif.shift(1))).to_numpy()


def perf(nav):
    nav = np.asarray(nav, float)
    if len(nav) < 2 or nav[0] == 0: return None
    years = len(nav) / 252; tot = nav[-1] / nav[0] - 1
    cagr = (1 + tot) ** (1 / years) - 1
    peak = np.maximum.accumulate(nav); dd = (nav - peak) / peak
    r = np.diff(nav) / nav[:-1]
    return dict(cagr=round(100*cagr,2), dd=round(100*dd.min(),2),
               calmar=round(cagr/abs(dd.min()),2) if dd.min() else 0,
               sharpe=round(r.mean()/r.std()*np.sqrt(252),2) if r.std() else 0,
               total=round(100*tot,2))


def run():
    dates, didx, codes, close = load()
    T, K = close.shape
    ret = np.vstack([np.zeros((1, K)), close[1:] / close[:-1] - 1.0])
    ret = np.nan_to_num(ret)
    bull = np.column_stack([macd_bull(close[:, k]) for k in range(K)])
    avail = ~np.isnan(close)
    bull = bull & avail
    rs20 = np.full((T, K), -np.inf)
    rs20[20:] = close[20:] / close[:-20] - 1.0          # 20d momentum for ranking
    rs20[~avail] = -np.inf

    def weights(mode):
        W = np.zeros((T, K))
        for t in range(T):
            b = np.where(bull[t])[0]
            if len(b) == 0: continue
            if mode == "single300":
                if bull[t, 0]: W[t, 0] = 1.0
            elif mode == "ew_all":
                W[t, b] = 1.0 / len(b)
            elif mode == "top1":
                W[t, b[np.argmax(rs20[t, b])]] = 1.0
            elif mode == "top2":
                order = b[np.argsort(-rs20[t, b])][:2]
                W[t, order] = 1.0 / len(order)
        return W

    def backtest(W):
        Wl = np.vstack([np.zeros((1, K)), W[:-1]])   # T+1: act on prior-close decision
        port_ret = (Wl * ret).sum(axis=1)
        turn = np.abs(np.diff(Wl, axis=0, prepend=Wl[:1])).sum(axis=1)
        nav = np.cumprod(1 + port_ret - turn * COST)
        return nav, Wl.sum(axis=1)

    windows = {"IS 2015-2025": ("20150101", "20250101"), "OOS 2025-2026": ("20250101", "20260624")}
    strategies = {
        "B&H 沪深300": None,
        "单一300择时": "single300",
        "等权持有所有看多指数": "ew_all",
        "轮动 Top1(动量)": "top1",
        "轮动 Top2(动量)": "top2",
    }
    for wlabel, (s, e) in windows.items():
        si = next(i for i, d in enumerate(dates) if d >= s)
        ei = next((i for i, d in enumerate(dates) if d >= e), T - 1)
        print(f"\n{'='*86}\n{wlabel}\n{'='*86}")
        print(f"{'策略':<26}{'CAGR%':>7}{'回撤%':>8}{'Calmar':>8}{'Sharpe':>8}{'总收益%':>9}{'平均暴露%':>10}")
        for name, mode in strategies.items():
            if mode is None:
                nav = np.cumprod(1 + ret[si:ei+1, 0]); expo = 100
            else:
                navf, ex = backtest(weights(mode))
                nav = navf[si:ei+1] / navf[si]; expo = round(100*ex[si:ei+1].mean(),1)
            p = perf(nav)
            print(f"{name:<24}{p['cagr']:>7}{p['dd']:>8}{p['calmar']:>8}{p['sharpe']:>8}{p['total']:>9}{expo:>10}")


if __name__ == "__main__":
    run()
