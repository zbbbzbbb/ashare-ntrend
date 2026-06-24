"""
Fast, vectorized, fork-parallel research harness for the A-share N-trend strategy.

Faithfully reproduces the original 4-layer funnel (CSI300 MACD timing -> synthetic
industry rotation -> N-pattern stock selection -> sized execution with exits), but:
  * all signals are computed ONCE, vectorized across the whole [T x N] panel
  * the daily loop is pure numpy array indexing (no per-day DataFrame slicing)
  * config sweeps fan out across cores via fork (panels shared copy-on-write)

This lets us test improvement hypotheses (exit rule, capital utilization, regime,
profit-taking) in seconds instead of minutes per run.

Read-only against ~/quant-data/catalog.duckdb.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from dataclasses import dataclass, field

CATALOG = str(Path.home() / "quant-data" / "catalog.duckdb")

# ----- module-level globals populated by build_panels(); inherited by forked workers
DATES: list[str] = []           # YYYYMMDD
DIDX: dict[str, int] = {}
CLOSE = HIGH = LOW = VOL = None  # [T, N] float arrays
CSI_CLOSE = None                 # [T] benchmark close aligned to DATES
COLS: list[str] = []             # universe ts_codes (column order)
IND_COLS: dict[str, np.ndarray] = {}  # industry code -> array of column indices

# precomputed stock signal panels [T, N]
S_MACD_BULL = S_VOLOK = S_ABOVEBBI = S_TWODOWN = S_ISN = None
S_NQUAL = S_ATR = S_VALID = None
S_MA20 = S_MA50 = S_MA60 = S_VOLRATIO = None
S_LEG1 = S_PULL = S_BSTR = S_BBI = None   # N-pattern components + BBI value (for inspection)
# precomputed industry score panel [T, M] + the industry code order
IND_SCORE = None
IND_MACD_BULL = None
IND_CODE_ORDER: list[str] = []
IND_RSRANK = IND_VOLSCORE = IND_NQUAL = IND_RET20 = IND_CLOSE = None  # components (inspection)


def _q(con, sql, params=None):
    return con.execute(sql, params or []).fetchdf()


def build_panels(start="20140601", end="20260624", ref_date="2020-06-30",
                 universe_size=1000, min_mktcap=200000, verbose=True):
    """Load everything once into module globals (numpy panels + signals)."""
    global DATES, DIDX, CLOSE, HIGH, LOW, VOL, CSI_CLOSE, COLS, IND_COLS
    con = duckdb.connect(CATALOG, read_only=True)
    iso = lambda d: f"{d[:4]}-{d[4:6]}-{d[6:8]}" if "-" not in d else d
    ymd = lambda d: str(d)[:10].replace("-", "")

    # 1. calendar
    cal = _q(con, "SELECT cal_date FROM trade_cal WHERE is_open=1 AND cal_date>=? AND cal_date<=? ORDER BY cal_date",
             [iso(start), iso(end)])
    DATES = [ymd(d) for d in cal["cal_date"]]
    DIDX = {d: i for i, d in enumerate(DATES)}
    T = len(DATES)
    if verbose: print(f"[panels] {T} trading days {DATES[0]}..{DATES[-1]}")

    # 2. universe: top-N by circ_mv at ref date, exclude ST, listed >= 250d
    uni = _q(con, """
        SELECT d.ts_code
        FROM daily d
        LEFT JOIN stock_basic s ON d.ts_code=s.ts_code
        LEFT JOIN daily_basic b ON d.ts_code=b.ts_code AND d.trade_date=b.trade_date
        WHERE d.trade_date=? AND d.close>0 AND s.list_date IS NOT NULL
          AND s.name NOT LIKE '%ST%'
          AND DATEDIFF('day', s.list_date::DATE, d.trade_date::DATE) >= 250
          AND COALESCE(b.circ_mv,0) >= ?
        ORDER BY COALESCE(b.circ_mv,0) DESC
        LIMIT ?
    """, [ref_date, min_mktcap, universe_size])
    COLS = uni["ts_code"].tolist()
    N = len(COLS)
    cidx = {c: j for j, c in enumerate(COLS)}
    if verbose: print(f"[panels] universe {N} stocks (top by circ_mv @ {ref_date})")

    # 3. adjusted daily panel for universe
    ph = ",".join(["?"] * N)
    raw = _q(con, f"""
        SELECT d.ts_code, d.trade_date,
               d.high*COALESCE(a.adj_factor,1.0) AS h,
               d.low *COALESCE(a.adj_factor,1.0) AS l,
               d.close*COALESCE(a.adj_factor,1.0) AS c,
               d.vol AS v
        FROM daily d LEFT JOIN adj_factor a ON d.ts_code=a.ts_code AND d.trade_date=a.trade_date
        WHERE d.ts_code IN ({ph}) AND d.trade_date>=? AND d.trade_date<=?
    """, COLS + [iso(start), iso(end)])
    raw["di"] = raw["trade_date"].map(lambda d: DIDX.get(ymd(d), -1))
    raw["ci"] = raw["ts_code"].map(cidx)
    raw = raw[raw["di"] >= 0]
    CLOSE = np.full((T, N), np.nan); HIGH = np.full((T, N), np.nan)
    LOW = np.full((T, N), np.nan);   VOL = np.full((T, N), np.nan)
    di = raw["di"].to_numpy(); ci = raw["ci"].to_numpy()
    CLOSE[di, ci] = raw["c"].to_numpy(); HIGH[di, ci] = raw["h"].to_numpy()
    LOW[di, ci] = raw["l"].to_numpy();   VOL[di, ci] = raw["v"].to_numpy()
    if verbose: print(f"[panels] price panel {CLOSE.shape} filled")

    # 4. benchmark CSI300
    bench = _q(con, "SELECT trade_date, close FROM index_daily WHERE ts_code='000300.SH' AND trade_date>=? AND trade_date<=? ORDER BY trade_date",
               [iso(start), iso(end)])
    CSI_CLOSE = np.full(T, np.nan)
    for d, c in zip(bench["trade_date"], bench["close"]):
        i = DIDX.get(ymd(d), -1)
        if i >= 0: CSI_CLOSE[i] = c
    CSI_CLOSE = pd.Series(CSI_CLOSE).ffill().to_numpy()

    # 5. industry membership @ ref date, restricted to universe
    mem = _q(con, """
        SELECT l3_code, ts_code FROM index_member_all
        WHERE in_date<=? AND (out_date IS NULL OR out_date>?)
    """, [ref_date, ref_date])
    IND_COLS = {}
    for code, g in mem.groupby("l3_code"):
        idxs = [cidx[t] for t in g["ts_code"] if t in cidx]
        if len(idxs) >= 3:
            IND_COLS[code] = np.array(sorted(idxs))
    if verbose: print(f"[panels] industries with >=3 universe members: {len(IND_COLS)}")
    con.close()


# ---------- vectorized signal helpers (operate on [T,N] DataFrames) ----------
def _ema(df, span): return df.ewm(span=span, adjust=False).mean()


def compute_stock_signals(macd_fast=12, macd_slow=26, macd_signal=9,
                          n_l1s=20, n_l1e=10, n_pe=5, verbose=True):
    """Populate S_* signal panels for the stock universe."""
    global S_MACD_BULL, S_VOLOK, S_ABOVEBBI, S_TWODOWN, S_ISN, S_NQUAL, S_ATR
    global S_VALID, S_MA20, S_MA50
    c = pd.DataFrame(CLOSE); h = pd.DataFrame(HIGH); l = pd.DataFrame(LOW); v = pd.DataFrame(VOL)

    # MACD bullish (DIF>DEA)
    dif = _ema(c, macd_fast) - _ema(c, macd_slow)
    dea = _ema(dif, macd_signal)
    S_MACD_BULL = (dif > dea).to_numpy()

    # volume > 20d avg
    S_VOLOK = (v > v.rolling(20).mean()).to_numpy()

    # BBI
    bbi = (c.rolling(3).mean() + c.rolling(6).mean() + c.rolling(12).mean() + c.rolling(24).mean()) / 4
    below = c < bbi
    S_ABOVEBBI = (c > bbi).to_numpy()
    S_TWODOWN = (below & below.shift(1)).to_numpy()
    global S_BBI; S_BBI = bbi.to_numpy()

    # N-pattern + quality
    leg1 = c.shift(n_l1e) / c.shift(n_l1s) - 1.0
    pull = c.shift(n_pe) / c.shift(n_l1e) - 1.0
    rhigh = h.rolling(n_l1e + 1, min_periods=1).max().shift(1)
    bstr = c / rhigh - 1.0
    isn = (leg1 > 0.02) & (pull < -0.01) & (c > rhigh) & (bstr > 0.0)
    S_ISN = isn.to_numpy()
    global S_LEG1, S_PULL, S_BSTR
    S_LEG1 = leg1.to_numpy(); S_PULL = pull.to_numpy(); S_BSTR = bstr.to_numpy()
    leg1n = np.clip((leg1.to_numpy() - 0.02) / 0.10, 0, 1)
    pulln = np.clip(1 - (pull.to_numpy() / -0.01), 0, 1)
    bstrn = np.clip(bstr.to_numpy() / 0.03, 0, 1)
    q = (leg1n + pulln + bstrn) / 3.0
    q[~S_ISN] = 0.0
    S_NQUAL = np.nan_to_num(q)

    # ATR(14) Wilder — elementwise true range across the [T,N] panel
    prev_c = c.shift(1)
    tr = np.maximum.reduce([(h - l).to_numpy(),
                            (h - prev_c).abs().to_numpy(),
                            (l - prev_c).abs().to_numpy()])
    S_ATR = pd.DataFrame(tr).ewm(alpha=1/14, adjust=False).mean().to_numpy()

    # moving averages for MA-trailing exits + trend filter
    S_MA20 = c.rolling(20).mean().to_numpy()
    S_MA50 = c.rolling(50).mean().to_numpy()
    global S_MA60, S_VOLRATIO
    S_MA60 = c.rolling(60).mean().to_numpy()
    S_VOLRATIO = (v / v.rolling(20).mean()).to_numpy()   # volume surge ratio

    # valid = has >=250 observations so far AND price present today
    obs = c.notna().cumsum().to_numpy()
    S_VALID = (obs >= 250) & (~np.isnan(CLOSE))
    if verbose: print(f"[signals] stock signals ready macd=({macd_fast},{macd_slow},{macd_signal})")


def compute_industry_scores(macd_fast=12, macd_slow=26, macd_signal=9,
                            n_l1s=20, n_l1e=10, n_pe=5, rs_ascending=True, verbose=True):
    """Build synthetic equal-weight industry composites and score panel.

    rs_ascending=True  -> favor momentum LEADERS (the brief's intent / my default)
    rs_ascending=False -> reproduce the ORIGINAL code's inverted rank (favors laggards)
    """
    global IND_SCORE, IND_MACD_BULL, IND_CODE_ORDER
    codes = sorted(IND_COLS.keys())
    IND_CODE_ORDER = codes
    T = len(DATES); M = len(codes)
    icl = np.full((T, M), np.nan); ivl = np.full((T, M), np.nan)
    for m, code in enumerate(codes):
        cols = IND_COLS[code]
        icl[:, m] = np.nanmean(CLOSE[:, cols], axis=1)
        ivl[:, m] = np.nansum(VOL[:, cols], axis=1)
    ic = pd.DataFrame(icl); iv = pd.DataFrame(ivl)

    dif = _ema(ic, macd_fast) - _ema(ic, macd_slow)
    dea = _ema(dif, macd_signal)
    macd_bull = (dif > dea)
    IND_MACD_BULL = macd_bull.to_numpy()

    ret20 = (ic / ic.shift(20) - 1.0)
    rs_rank = ret20.rank(axis=1, pct=True, ascending=rs_ascending)
    vol_ratio = iv / iv.rolling(60).mean()
    vol_score = (vol_ratio / 2.0).clip(upper=1.0)

    # n-quality on composite
    leg1 = ic.shift(n_l1e) / ic.shift(n_l1s) - 1.0
    pull = ic.shift(n_pe) / ic.shift(n_l1e) - 1.0
    rhigh = ic.rolling(n_l1e + 1, min_periods=1).max().shift(1)  # composite has no real high
    bstr = ic / rhigh - 1.0
    isn = (leg1 > 0.02) & (pull < -0.01) & (ic > rhigh)
    leg1n = np.clip((leg1.to_numpy() - 0.02) / 0.10, 0, 1)
    pulln = np.clip(1 - (pull.to_numpy() / -0.01), 0, 1)
    bstrn = np.clip(bstr.to_numpy() / 0.03, 0, 1)
    nq = (leg1n + pulln + bstrn) / 3.0
    nq[~isn.to_numpy()] = 0.0
    nq = np.nan_to_num(nq)

    score = (macd_bull.to_numpy().astype(float) * 0.25
             + np.nan_to_num(rs_rank.to_numpy()) * 0.35
             + np.nan_to_num(vol_score.to_numpy()) * 0.20
             + nq * 0.20)
    IND_SCORE = score
    global IND_RSRANK, IND_VOLSCORE, IND_NQUAL, IND_RET20, IND_CLOSE
    IND_RSRANK = rs_rank.to_numpy(); IND_VOLSCORE = vol_score.to_numpy()
    IND_NQUAL = nq; IND_RET20 = ret20.to_numpy(); IND_CLOSE = icl
    if verbose: print(f"[signals] industry scores ready ({M} industries)")


def regime_series(full_bullish=True, macd_fast=12, macd_slow=26, macd_signal=9):
    """Benchmark regime gate over DATES. full_bullish replicates original
    (DIF>DEA & hist expanding & DIF rising); else relaxed (DIF>DEA)."""
    c = pd.Series(CSI_CLOSE)
    dif = c.ewm(span=macd_fast, adjust=False).mean() - c.ewm(span=macd_slow, adjust=False).mean()
    dea = dif.ewm(span=macd_signal, adjust=False).mean()
    hist = (dif - dea) * 2
    bull = dif > dea
    if full_bullish:
        bull = bull & (hist.abs() > hist.abs().shift(1)) & (dif > dif.shift(1))
    return bull.to_numpy()


# ----------------------------- backtest config ------------------------------
@dataclass
class Config:
    name: str = "baseline"
    start: str = "20150101"
    end: str = "20250101"
    capital: float = 1_500_000.0
    top_industries: int = 5
    max_positions: int = 10
    max_weekly_buys: int = 2
    pos_pct: float = 0.15                # fraction of capital per position
    # regime
    full_bullish: bool = True            # True=original 3-condition; False=DIF>DEA only
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # exit
    exit_rule: str = "bbi2"              # bbi2 | chandelier | ma_trail | time_stop | swing
    chandelier_k: float = 3.0
    ma_trail: int = 20
    hold_days: int = 10                  # for time_stop / swing time-exit
    take_profit: float = 0.0             # swing: exit when pnl >= this (0=off)
    # entry quality filters (0/1.0/True = off)
    vol_mult: float = 1.0                # require vol >= vol_mult x 20d-avg volume
    entry_trend_ma: int = 0              # require close > MA(n); 0=off (supports 20/50/60)
    relax_n: bool = True                 # if False, ALWAYS require a real N-pattern
    hard_stop: float = -0.10             # catastrophe stop for trailing rules (frac); None to disable
    # profit taking
    profit_take: bool = True             # +20% -> sell 50%
    profit_threshold: float = 0.20
    profit_fraction: float = 0.50
    # costs
    comm: float = 0.00025
    stamp: float = 0.0005
    slippage: float = 0.0               # 0 reproduces original (slip/impact never applied)
    impact: float = 0.0
    cost_stress: float = 1.0


def run_backtest(cfg: Config) -> dict:
    """Event-driven loop over precomputed panels. Returns metrics + trade log."""
    s = DIDX.get(cfg.start);  e = DIDX.get(cfg.end)
    if s is None:  # nearest >= start
        s = next(i for i, d in enumerate(DATES) if d >= cfg.start)
    if e is None:
        e = next((i for i, d in enumerate(DATES) if d >= cfg.end), len(DATES) - 1)
    regime = regime_series(cfg.full_bullish, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)

    buy_rate = (cfg.comm + cfg.slippage + cfg.impact) * cfg.cost_stress
    sell_rate = (cfg.comm + cfg.stamp + cfg.slippage + cfg.impact) * cfg.cost_stress

    cash = cfg.capital
    positions: dict[int, dict] = {}   # col -> {shares, entry_px, entry_di, above_bbi, hi_close, pt_done}
    nav_hist = []
    trades = []                       # closed round-trip pnl records
    week = None; weekly_buys = 0
    warmup = max(s, 60)

    M = len(IND_CODE_ORDER)

    for di in range(warmup, e + 1):
        wk = DATES[di][:6]
        if wk != week:
            week = wk; weekly_buys = 0

        px = CLOSE[di]

        # ---- 1. exits ----
        for j in list(positions.keys()):
            p = px[j]
            if np.isnan(p):
                continue
            pos = positions[j]
            if p > pos["hi_close"]:
                pos["hi_close"] = p
            pnl = p / pos["entry_px"] - 1.0
            exit_now = False; reason = ""

            if cfg.exit_rule == "bbi2":
                if pos["above_bbi"]:
                    if S_TWODOWN[di, j]:
                        exit_now, reason = True, "bbi_exit"
                else:
                    if pnl <= -0.03:
                        exit_now, reason = True, "stop_3pct"
            elif cfg.exit_rule == "chandelier":
                atr = S_ATR[di, j]
                if not np.isnan(atr) and p < pos["hi_close"] - cfg.chandelier_k * atr:
                    exit_now, reason = True, "chandelier"
                elif cfg.hard_stop is not None and pnl <= cfg.hard_stop:
                    exit_now, reason = True, "hard_stop"
            elif cfg.exit_rule == "ma_trail":
                ma = S_MA20[di, j] if cfg.ma_trail == 20 else S_MA50[di, j]
                if not np.isnan(ma) and p < ma:
                    exit_now, reason = True, "ma_break"
                elif cfg.hard_stop is not None and pnl <= cfg.hard_stop:
                    exit_now, reason = True, "hard_stop"
            elif cfg.exit_rule == "time_stop":
                # harvest the short-lived momentum pop, then get out
                if di - pos["entry_di"] >= cfg.hold_days:
                    exit_now, reason = True, "time_stop"
                elif cfg.hard_stop is not None and pnl <= cfg.hard_stop:
                    exit_now, reason = True, "hard_stop"
            elif cfg.exit_rule == "swing":
                # take-profit OR catastrophe stop OR time stop (whichever first)
                if cfg.take_profit > 0 and pnl >= cfg.take_profit:
                    exit_now, reason = True, "take_profit"
                elif cfg.hard_stop is not None and pnl <= cfg.hard_stop:
                    exit_now, reason = True, "hard_stop"
                elif di - pos["entry_di"] >= cfg.hold_days:
                    exit_now, reason = True, "time_stop"

            if exit_now:
                cash += p * pos["shares"] * (1 - sell_rate)
                trades.append({"col": j, "ts": COLS[j], "entry_di": pos["entry_di"],
                               "exit_di": di, "ret": pnl, "reason": reason,
                               "hold": di - pos["entry_di"], "shares": pos["shares"],
                               "entry_px": pos["entry_px"], "exit_px": p})
                del positions[j]
                continue

            # profit taking (partial)
            if cfg.profit_take and not pos["pt_done"] and pnl >= cfg.profit_threshold:
                sell_sh = int(pos["shares"] * cfg.profit_fraction)
                if sell_sh > 0:
                    cash += p * sell_sh * (1 - sell_rate)
                    trades.append({"col": j, "ts": COLS[j], "entry_di": pos["entry_di"],
                                   "exit_di": di, "ret": pnl, "reason": "profit_take",
                                   "hold": di - pos["entry_di"], "shares": sell_sh,
                                   "entry_px": pos["entry_px"], "exit_px": p})
                    pos["shares"] -= sell_sh
                    pos["pt_done"] = True

        # ---- 2. entries ----
        if regime[di] and weekly_buys < cfg.max_weekly_buys and len(positions) < cfg.max_positions:
            sc = IND_SCORE[di]
            mb = IND_MACD_BULL[di]
            order = np.argsort(-np.nan_to_num(sc, nan=-1e9))
            bull_top = [m for m in order if mb[m]][:cfg.top_industries]
            if len(bull_top) < cfg.top_industries:
                seen = set(bull_top)
                for m in order:
                    if m not in seen:
                        bull_top.append(m)
                        if len(bull_top) >= cfg.top_industries:
                            break
            cand = set()
            for m in bull_top[:cfg.top_industries]:
                cand.update(IND_COLS[IND_CODE_ORDER[m]].tolist())
            ma_panel = {20: S_MA20, 50: S_MA50, 60: S_MA60}.get(cfg.entry_trend_ma)
            cand = [j for j in cand if S_VALID[di, j] and j not in positions
                    and S_MACD_BULL[di, j] and S_VOLRATIO[di, j] >= cfg.vol_mult
                    and (ma_panel is None or px[j] > ma_panel[di, j])]
            if cand:
                n_qual = [j for j in cand if S_ISN[di, j]]
                if cfg.relax_n:
                    pool = n_qual if len(n_qual) >= 3 else cand
                else:
                    pool = n_qual            # always require a real N-pattern
                pool.sort(key=lambda j: (S_NQUAL[di, j] * 0.30 + 0.25 * S_MACD_BULL[di, j]
                                         + 0.20 * S_VOLOK[di, j] + 0.15 * S_ABOVEBBI[di, j]
                                         + 0.10 * S_ISN[di, j]), reverse=True)
                for j in pool:
                    if weekly_buys >= cfg.max_weekly_buys or len(positions) >= cfg.max_positions:
                        break
                    p = px[j]
                    if np.isnan(p) or p <= 0:
                        continue
                    target = cfg.capital * cfg.pos_pct
                    shares = int(target / p // 100) * 100
                    cost = p * shares * (1 + buy_rate)
                    if shares <= 0:
                        continue
                    if cost > cash:
                        shares = int(cash / (p * (1 + buy_rate)) // 100) * 100
                        if shares <= 0:
                            continue
                        cost = p * shares * (1 + buy_rate)
                    cash -= cost
                    positions[j] = {"shares": shares, "entry_px": p, "entry_di": di,
                                    "above_bbi": bool(S_ABOVEBBI[di, j]), "hi_close": p,
                                    "pt_done": False}
                    weekly_buys += 1

        # ---- 3. NAV ----
        pv = 0.0
        for j, pos in positions.items():
            p = px[j]
            pv += (p if not np.isnan(p) else pos["entry_px"]) * pos["shares"]
        nav_hist.append(cash + pv)

    return _metrics(cfg, nav_hist, trades)


def _metrics(cfg, nav_hist, trades):
    nav = np.array(nav_hist, dtype=float)
    if len(nav) < 2:
        return {"name": cfg.name, "error": "no_nav"}
    years = len(nav) / 252
    total_ret = nav[-1] / cfg.capital - 1
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    maxdd = dd.min()
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    rets = np.diff(nav) / nav[:-1]
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0

    rt = [t["ret"] for t in trades]
    wins = [r for r in rt if r > 0]; losses = [r for r in rt if r <= 0]
    # pnl in yuan for profit factor
    pnl_y = [(t["exit_px"] - t["entry_px"]) * t["shares"] for t in trades]
    gp = sum(p for p in pnl_y if p > 0); gl = abs(sum(p for p in pnl_y if p < 0))
    pf = gp / gl if gl > 0 else float("inf")
    holds = [t["hold"] for t in trades]
    win_holds = [t["hold"] for t in trades if t["ret"] > 0]

    # invested fraction proxy: avg positions over time isn't tracked; use trade count
    return {
        "name": cfg.name,
        "cagr_pct": round(cagr * 100, 2),
        "total_ret_pct": round(total_ret * 100, 2),
        "max_dd_pct": round(maxdd * 100, 2),
        "calmar": round(calmar, 2),
        "sharpe": round(sharpe, 2),
        "profit_factor": round(pf, 2) if pf != float("inf") else 999,
        "win_rate_pct": round(100 * len(wins) / len(rt), 1) if rt else 0,
        "n_trades": len(trades),
        "avg_hold": round(np.mean(holds), 1) if holds else 0,
        "avg_win_hold": round(np.mean(win_holds), 1) if win_holds else 0,
        "avg_win_pct": round(100 * np.mean(wins), 2) if wins else 0,
        "avg_loss_pct": round(100 * np.mean(losses), 2) if losses else 0,
        "best_pct": round(100 * max(rt), 1) if rt else 0,
        "final_nav": round(nav[-1]),
        "years": round(years, 2),
    }
