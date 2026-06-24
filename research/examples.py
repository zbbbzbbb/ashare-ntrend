"""
Generate concrete, fully-exposed worked examples of the strategy's decisions, so
they can be investigated by hand:
  - market timing state on the day
  - industry selection (all component scores, which top-5 were chosen)
  - stock selection (per-stock signal breakdown, hard filters, the pick)
  - the forward outcome of the pick (5/10/20-day return)

Picks real bullish-regime entry days that turned out WELL and BADLY, so the
failure modes are visible. Writes a Markdown file.
"""
import warnings
import numpy as np, pandas as pd, duckdb
from pathlib import Path
warnings.filterwarnings("ignore"); np.seterr(all="ignore")

import research.fast_bt as fb

CATALOG = str(Path.home() / "quant-data" / "catalog.duckdb")
OUT = Path(__file__).parent.parent / "output" / "选股示例_供研究.md"
TOPI = 5


def load_names():
    con = duckdb.connect(CATALOG, read_only=True)
    sn = {r[0]: r[1] for r in con.execute("SELECT ts_code, name FROM stock_basic").fetchall()}
    inm = {r[0]: r[1] for r in con.execute("SELECT index_code, industry_name FROM index_classify WHERE level='L3'").fetchall()}
    con.close()
    return sn, inm


def fwd(di, j, k):
    if di + k >= fb.CLOSE.shape[0]:
        return None
    a, b = fb.CLOSE[di, j], fb.CLOSE[di + k, j]
    return (b / a - 1.0) if (not np.isnan(a) and not np.isnan(b) and a > 0) else None


def top_industries(di):
    sc = fb.IND_SCORE[di]; mb = fb.IND_MACD_BULL[di]
    order = np.argsort(-np.nan_to_num(sc, nan=-1e9))
    top = [m for m in order if mb[m]][:TOPI]
    if len(top) < TOPI:
        for m in order:
            if m not in top:
                top.append(m)
                if len(top) >= TOPI: break
    return top[:TOPI]


def candidate_pool(di):
    """Reproduce the entry funnel for day di -> (selected industry idxs, ranked stock pool)."""
    inds = top_industries(di)
    cand = set()
    for m in inds:
        cand.update(fb.IND_COLS[fb.IND_CODE_ORDER[m]].tolist())
    cand = [j for j in cand if fb.S_VALID[di, j] and fb.S_MACD_BULL[di, j] and (fb.S_VOLRATIO[di, j] >= 1.0)]
    n_qual = [j for j in cand if fb.S_ISN[di, j]]
    pool = n_qual if len(n_qual) >= 3 else cand
    score = lambda j: (fb.S_NQUAL[di, j]*0.30 + 0.25*fb.S_MACD_BULL[di, j] + 0.20*fb.S_VOLOK[di, j]
                       + 0.15*fb.S_ABOVEBBI[di, j] + 0.10*fb.S_ISN[di, j])
    pool = sorted(pool, key=score, reverse=True)
    return inds, cand, pool, score


def scan_for_examples():
    """Find bullish-regime days with a top pick, label by 20d forward outcome."""
    regime = fb.regime_series(full_bullish=True)
    s = next(i for i, d in enumerate(fb.DATES) if d >= "20160101")
    e = next(i for i, d in enumerate(fb.DATES) if d >= "20250101")
    events = []
    last = -10
    for di in range(s, e):
        if not regime[di] or di - last < 15:
            continue
        _, _, pool, _ = candidate_pool(di)
        if not pool:
            continue
        j = pool[0]; f20 = fwd(di, j, 20)
        if f20 is None:
            continue
        events.append((di, j, f20)); last = di
    events.sort(key=lambda x: x[2])
    losers = events[:2]
    winners = events[-2:]
    return winners, losers


def md_industries(di, inds, inm):
    sc, mb = fb.IND_SCORE[di], fb.IND_MACD_BULL[di]
    order = np.argsort(-np.nan_to_num(sc, nan=-1e9))[:10]
    rows = ["| 行业 | MACD看多 | 20日涨幅 | 相对强度排名 | 放量分 | N型质量 | **综合分** | 选中? |",
            "|---|---|---|---|---|---|---|---|"]
    for m in order:
        code = fb.IND_CODE_ORDER[m]
        nm = inm.get(code, code)
        sel = "✅" if m in inds else ""
        rows.append(f"| {nm}({code}) | {'是' if mb[m] else '否'} | {fb.IND_RET20[di,m]*100:.1f}% | "
                    f"{fb.IND_RSRANK[di,m]:.2f} | {fb.IND_VOLSCORE[di,m]:.2f} | {fb.IND_NQUAL[di,m]:.2f} | "
                    f"**{sc[m]:.3f}** | {sel} |")
    return "\n".join(rows)


def md_stocks(di, pool, score_fn, sn, highlight):
    rows = ["| 代码 | 名称 | MACD看多 | 量比(/20日) | N型? | 第一段涨 | 回调 | 突破幅度 | N质量 | BBI上方 | **打分** | 买? |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for rank, j in enumerate(pool[:12]):
        ts = fb.COLS[j]
        buy = "🔵买入" if j == highlight else ("①" if rank == 0 else "")
        rows.append(f"| {ts} | {sn.get(ts,'?')} | {'是' if fb.S_MACD_BULL[di,j] else '否'} | "
                    f"{fb.S_VOLRATIO[di,j]:.2f} | {'是' if fb.S_ISN[di,j] else '否'} | "
                    f"{fb.S_LEG1[di,j]*100:.1f}% | {fb.S_PULL[di,j]*100:.1f}% | {fb.S_BSTR[di,j]*100:.1f}% | "
                    f"{fb.S_NQUAL[di,j]:.2f} | {'是' if fb.S_ABOVEBBI[di,j] else '否'} | **{score_fn(j):.3f}** | {buy} |")
    return "\n".join(rows)


def explain(di, j_pick, sn, inm, tag):
    d = fb.DATES[di]
    inds, cand, pool, score_fn = candidate_pool(di)
    # market timing detail (CSI300)
    c = pd.Series(fb.CSI_CLOSE)
    dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean(); hist = (dif - dea) * 2
    f5, f10, f20 = fwd(di, j_pick, 5), fwd(di, j_pick, 10), fwd(di, j_pick, 20)
    pct = lambda x: f"{x*100:+.1f}%" if x is not None else "n/a"
    ts = fb.COLS[j_pick]
    out = [f"\n---\n\n## {tag}：{d}  买入 {sn.get(ts,'?')}（{ts}）\n",
           "### ① 大盘择时（沪深300 MACD）",
           f"- DIF={dif.iloc[di]:.1f}  DEA={dea.iloc[di]:.1f}  柱={hist.iloc[di]:.1f}  "
           f"DIF>DEA={'是' if dif.iloc[di]>dea.iloc[di] else '否'}  柱扩张={'是' if abs(hist.iloc[di])>abs(hist.iloc[di-1]) else '否'}  "
           f"DIF上行={'是' if dif.iloc[di]>dif.iloc[di-1] else '否'}",
           f"- **结论：牛市，允许开仓**\n",
           "### ② 行业选择（按综合分排序，前10；✅=选中的前5）",
           md_industries(di, inds, inm),
           f"\n> 选股范围 = 这5个行业的成分股，共 {len(cand)} 只通过硬性过滤（MACD看多 且 放量）\n",
           "### ③ 个股选择（候选池打分排序，🔵=实际买入）",
           md_stocks(di, pool, score_fn, sn, j_pick),
           f"\n### ④ 这笔交易的真实结果（买入价 {fb.CLOSE[di,j_pick]:.2f}）",
           f"- 5日后 {pct(f5)} ｜ 10日后 {pct(f10)} ｜ 20日后 {pct(f20)}",
           f"- **{'✅ 成功' if (f20 or 0)>0 else '❌ 失败'}（20日 {pct(f20)}）**"]
    return "\n".join(out)


def main():
    fb.build_panels(verbose=False)
    fb.compute_stock_signals(12, 26, 9, verbose=False)
    fb.compute_industry_scores(12, 26, 9, rs_ascending=True, verbose=False)
    sn, inm = load_names()
    winners, losers = scan_for_examples()

    parts = ["# A股趋势策略 · 选股决策示例（供人工研究）",
             "",
             "> 用途：把策略每一步的**真实信号数值**摊开，方便你逐个排查"
             "“为什么选这个行业/这只票”“信号哪里失灵”。",
             "> 参数：MACD(12,26)择时；行业用成分股等权合成、按 "
             "`0.25·MACD + 0.35·相对强度排名 + 0.20·放量 + 0.20·N型质量` 打分（已修正"
             "原代码把动量排序方向写反的bug，这里是“选龙头”的正确方向）。",
             "> 个股打分：`0.30·N型质量 + 0.25·MACD + 0.20·放量 + 0.15·BBI上方 + 0.10·N型`；"
             "硬过滤：MACD看多 **且** 放量。",
             "> N型定义：第一段(T-20→T-10)涨>2%、回调(T-10→T-5)跌>1%、今日收盘突破近11日高点。",
             "",
             "## 两个“成功”示例（信号奏效）"]
    for di, j, f20 in reversed(winners):
        parts.append(explain(di, j, sn, inm, "成功示例"))
    parts.append("\n## 两个“失败”示例（信号失灵——重点研究这里）")
    for di, j, f20 in losers:
        parts.append(explain(di, j, sn, inm, "失败示例"))

    parts += ["\n---\n", "## 怎么用这些示例去改进",
              "1. **看失败示例的②行业表**：被选中的行业，相对强度排名/N型质量是不是其实很弱？"
              "（综合分里 0.25 的 MACD 是 0/1 离散，常常靠它凑数）",
              "2. **看③个股表的“第一段涨/回调/突破幅度”**：失败的票往往是“勉强达标”——第一段只涨"
              "2%出头、突破幅度<1%。可以加严阈值（如第一段>5%、突破>2%）看胜率是否提升。",
              "3. **看“放量量比”**：很多失败票量比刚过1.0。可试 1.5 或 2.0 的更强放量要求。",
              "4. **看BBI上方**：入场时在BBI下方的，后续表现是否更差？",
              "5. **对照真实结果**：把5/10/20日收益拉出来，验证我们的发现——edge集中在5-10日，"
              "之后衰减。",
              "",
              "复现：`python3 -m research.examples`（数据只读 `~/quant-data/catalog.duckdb`）"]
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"written -> {OUT}")
    print(f"winners: {[(fb.DATES[d], fb.COLS[j], round(f*100,1)) for d,j,f in winners]}")
    print(f"losers : {[(fb.DATES[d], fb.COLS[j], round(f*100,1)) for d,j,f in losers]}")


if __name__ == "__main__":
    main()
