# A-share Top-Down Strategy — Research Brief (v2, method-agnostic)

(Standalone brief for a NEW, independent strategy project. Execute in a fresh session.
This file is self-contained: every rule needed lives here. Read it fully, then begin.)

(The TOP-DOWN STRUCTURE is fixed; the SPECIFIC METHODS are deliberately left open.
Do NOT assume any particular indicator — no MACD, no N-pattern, no BBI, no fixed
thresholds are prescribed. Which signals to use, and how to time / rotate / select,
are open questions YOU must answer with evidence.)

---

# 0. OBJECTIVE

You are an experienced quantitative researcher and engineer.

Build and validate a long-only A-share **top-down** strategy whose objective is:

**Maximize robust out-of-sample long-term compound return through correct regime
participation and leadership concentration, while keeping drawdown disciplined
(this is a capital-preservation mandate, not a deep-drawdown one — see §2).**

Structured as three layers, applied top-down:

1. Market Timing — decide when to take risk and how much
2. Industry / Sector Rotation — concentrate capital into the leading direction(s)
3. Stock / Instrument Selection — pick what to hold within that direction

Working hypothesis: A-share excess return comes mainly from correct regime
identification, sector leadership rotation, trend participation/persistence, and
capital concentration — NOT from accounting factors, statistical arbitrage,
mean-reversion arb, or black-box ML.

> IMPORTANT: the framework above is a HYPOTHESIS to be tested, not a known result.
> A first-class duty is to PROVE (or disprove) that the three-layer top-down structure
> beats naive benchmarks (§8). If a layer adds no value, say so. Do not target-chase
> a return number (§9).

---

# 1. HARD CONSTRAINTS

## Independence / workspace isolation (HARD)
Derive ALL strategy logic, factors, and signals INDEPENDENTLY within THIS folder. Do
NOT read, reuse, adapt, or reference code / logic / notes / factors / findings from any
other project on this machine. You MAY use ONLY: (a) the shared read-only market data
lake, and (b) a data-facts verification ledger (see §6). General research METHODOLOGY
is standard practice, not a sibling finding — applying it is fine.

## Long-only
Allowed vehicles: A-share stocks, listed ETFs (broad / sector / thematic / bond /
money-market), cash. Any mix on any bar. NOT allowed: shorting, leverage, margin,
derivatives, leveraged/inverse ETFs, QDII.

> A SIGNAL and the INSTRUMENT used to EXPRESS it are two separate choices. A bullish
> timing view can be expressed via single stocks OR via an index/sector ETF — which is
> best is an EMPIRICAL question. Do NOT assume you must make money by picking single
> stocks.

## Permitted data
Adjusted price / volume / volatility / returns; **cash** dividends; size / valuation
ratios (market cap, PB, PE/EP from `daily_basic`); liquidity / turnover / Amihud;
market-structure flows (`hk_hold` northbound, `margin`); index / ETF / breadth
internals; macro / rates (`shibor`, `yc_cb`).
**No financial-statement fundamentals** (ROE, margins, accruals, growth, earnings-
statement items) — A-share statement data is fraud-prone and out of scope.

---

# 2. RISK PREFERENCE (this is where this mandate DIFFERS from a high-return mandate)

This is a **drawdown-controlled** mandate. Target Max Drawdown ≤ 35% (preferred ≤ 30%).
Reduce drawdown via timing and exits, not via excessive diversification. Strategies
needing persistent drawdowns > 50% are not preferred. Do NOT pursue deep-drawdown,
micro-cap-tail, or "return is the only maximand" designs — those belong to a different
mandate. Here, robust compounding WITHIN the drawdown budget is the goal.

---

# 3. CORE STRATEGY FRAMEWORK (structure fixed, methods open)

The STRUCTURE is fixed. The METHODS inside each step are OPEN — choose and justify with
evidence. Every "Research Questions" block is the actual work. No prescribed indicators.

## STEP 1 — MARKET TIMING
Open design space (choose, combine, COMPARE): which index/indices represent "the market"
(broad vs large/small cap vs all-market vs blend); what classifies the regime (trend,
momentum, volatility, market breadth, capital flows e.g. northbound/margin, valuation
percentile, macro/rates — free to mix); binary (in/out) vs continuous (exposure dial).
Research Questions: best regime definition (OOS, cost-aware, risk-adjusted); best index /
expression; robustness across cycles AND across parameter choices.

## STEP 2 — INDUSTRY / SECTOR ROTATION
Open: which taxonomy (SW L1/L2/L3, CSI industries, thematic baskets); how to measure
leadership / incremental inflow (relative strength, momentum, volume/turnover expansion,
northbound/margin net inflow, capital concentration); how many sectors; concentration vs
diversification; rotation cadence.
DATA REALITY (verify first): SW industry indices (801xxx/850xxx) have NO daily series in
this lake — synthesize from constituents, OR substitute CSI/thematic indices that DO have
real daily data. Check before designing.
Research Questions: best leadership/inflow proxy; right concentration & cadence;
**marginal value of this layer (mandatory) — how much does it add ON TOP of timing alone?
If nothing, report it.**

## STEP 3 — STOCK / INSTRUMENT SELECTION
Universe: constituents of selected sectors and/or relevant ETFs. Exclude ST, *ST, listed
< 250 trading days, delisting risk, extremely illiquid names.
Open: what selects a name (trend structure, breakout, relative strength, volume/price,
capital flows, liquidity, leadership) — design freely and COMPARE several rankings; how to
build a score that TRULY discriminates (not many ties).
Research Questions: best selection/ranking (head-to-head, OOS); **edge time-scale
(mandatory) — for any entry signal run an EVENT STUDY first: forward 5/10/20/40-day returns
vs a matched investable baseline; where is the edge, when does it decay, is the MEDIAN
positive? Design the holding period AROUND this.**

## STEP 4 — ENTRY (open)
Quantify the forward-return profile (event study, Step 3) before committing to any entry
rule, so entry/exit horizon matches the real edge horizon. Check whether the entry beats
simply holding the sector/index ETF.

## STEP 5 — RISK MANAGEMENT / EXIT (open)
Trailing / volatility-based / structure-based / time-based — your choice. Constraint:
the exit horizon MUST match the measured edge half-life (Step 3). Test stop robustness
across regimes.

## STEP 6 — PROFIT TAKING (research output, not a preset)
Derive scaling rules from the return distribution and edge time-scale; compare variants
(let-it-run vs partial-take vs time-exit), OOS, net of cost.

## STEP 7 — FREQUENCY & SIZING (research outputs)
Trade frequency, max concurrent positions, position sizing — derive them; no fixed cap
assumed. Balance edge capture vs turnover/cost. EOD review; state a look-ahead-free
decision/execution convention.

---

# 4. CAPITAL & INSTRUMENTS
Capital **1,500,000 RMB**. At this size small-caps and most listed ETFs are tradable, so
the size/liquidity premium is available; no institutional-capacity constraint. Avoid only
what you genuinely cannot fill (dead ETFs / extreme micro-caps below your liquidity floor).
Every result must be executable at 1.5M with modeled frictions, with a per-name ADV
participation cap + liquidity floor; report realized participation.

---

# 5. COSTS & SLIPPAGE (mandatory; ACTUALLY deducted from the equity curve)
- Stocks: buy ≈ 0.0096% (commission), sell ≈ 0.0596% (commission + stamp tax; stamp fell
  to ~2.5bp post-2023 — model both, report the conservative one). Commission floor ~5 RMB.
- ETFs: commission only both sides, no stamp (~0.5bp/side unless data says otherwise).
- Slippage + market impact ON TOP, rising with volatility / illiquidity / turnover /
  position size — model impact honestly (size / ADV / volatility-dependent). Small-cap,
  high-turnover, concentrated books carry materially higher friction.
- Run **Normal Cost AND ×2 Cost Stress** on every candidate. A result that only survives
  at low cost is REJECTED. No result is valid without full cost + slippage in the NAV.

---

# 6. DATA — read-only DuckDB lake at `~/quant-data/catalog.duckdb`
Partitioned Parquet under `lake/`, catalog `catalog.duckdb`, manifests under `_manifests/`.
History ~1990-12 → latest trade date (~2026-06); recent updates added datasets, not a
longer series, so genuinely-future holdout data does not yet exist. **Audit the full lake
in Phase 1.** Maintain a dated **data-facts verification ledger** inside this folder: when
you touch a dataset, verify and record what you touched (data facts ONLY — never strategy
content). Entries already verified may be taken as given.

Critical data discipline (verify, then rely):
- **Survivorship-free**: delisted names included with delist dates; ST/*ST included;
  index/fund typed separately from stocks.
- **Prices are RAW** — adjust via adj factors (hfq close = close × adj_factor). ALL
  signal/return series use ADJUSTED prices. ETFs adjust via `fund_adj`.
- **Units**: stock `daily` vol in 手 (×100 = shares), amount in 千元 (×1000 = RMB),
  pct_chg in PERCENT. Confirm fund/index units in audit.
- **Point-in-time**: use PIT index membership, listing/delisting dates, correct reporting
  lags; each date exclude not-yet-listed + suspended (absent rows) + apply realistic ST /
  price-limit handling.
- **ETF liquidity-onset (key)**: ETF history is shallow — broad-index ETFs (510300 CSI300,
  510500 CSI500) liquid only ~2012-13; a CSI1000 ETF only ~2022; most sector/thematic ETFs
  post-2015. Do NOT trade an ETF before it is liquid; for the early era fall back to the
  stock book and/or Total-Return indices as UNTRADABLE proxies.
- **Indices**: `index_daily` are PRICE indices (CSI300 000300.SH, CSI500 000905.SH,
  CSI1000 000852.SH, sector/style). `index_tr_cs` are TOTAL-RETURN (CSI300 h00300.CSI,
  CSI500 h00905.CSI — note NO CSI1000 TR exists). Use TR for benchmarking & pre-ETF proxy.
- **Idle cash**: if frequently in cash, credit it at the money-market rate (`shibor`) /
  hold a bond-or-money-market ETF — otherwise return and Sharpe are understated.
- Datasets include: daily, adj_factor, daily_basic, dividend, stock_basic, share_float,
  margin, hk_hold; fund_basic/fund_daily/fund_adj/fund_nav (ETFs); index_daily/index_tr_cs/
  index_classify/index_member_all/index_weight; shibor, yc_cb, trade_cal.

---

# 7. ANTI-OVERFIT & OUT-OF-SAMPLE DESIGN (the #1 guard — you may NOT spend it)
Opening the method space (multiple timing/rotation/selection methods) widens the search,
so a high backtest is more likely luck. Enforce from day one:

- **Pre-register** each factor/rule definition BEFORE seeing its result — ONE canonical
  definition per concept; do not cycle variants hunting the best fit.
- **Fix the validation design first; do NOT optimize the validator.**
- **Primary dev metric = out-of-sample COMPOUND RETURN** (median OOS CAGR; Calmar at the
  realized drawdown), via Walk-Forward / Combinatorial Purged CV with **purge + embargo ≥
  (max feature lookback + holding period)**. Report **concatenated OOS, never a single
  in-sample fit**.
- **Count EVERY configuration tried; pre-commit a trial budget; apply a Deflated-Sharpe-
  style haircut to the RETURN claim** (penalize trial count + skew/kurtosis). A result
  that fails the haircut is REJECTED.
- **Parameter plateau, not spikes.** An **economic mechanism is required** for every
  factor (risk / behavioral / microstructure); pure data-mined patterns are red flags.
- **Regime breakdown mandatory** (bull / bear / chop; and known A-share shocks). Return
  concentrated in one era or a few names is fragile — surface and discount it.
- **OOS holdout protocol (concrete):** reserve ONE terminal holdout = the most recent
  slice, chosen so it **spans ≥ 2 market regimes (do NOT validate on a single bull
  window — a single regime is misleading).** Lock it by date before research; spend it
  **exactly ONCE at the very end.** The instant you iterate on it, it is BURNED.
  Everything before runs on CPCV + deflated return + plateau + regime.
- **Per-layer attribution must also be checked ON the holdout**, not only in-sample.

---

# 8. BENCHMARKS & WHAT "VALUE" MEANS
Every candidate must clear ALL of:
1. **Beat CSI300 Total-Return (h00300.CSI) on ABSOLUTE compound return** (CAGR / terminal
   wealth) over the full sample. Beating only on Sharpe is insufficient. (Secondary refs:
   CSI500 TR; equal-weight investable universe.)
2. **Beat the naive top-down baseline** = a single broad-index ETF gated by your timing
   layer. If the full 3-layer system cannot beat this, the lower layers are a burden — say so.
3. **Per-layer attribution** — quantify the marginal OOS value of timing, then industry on
   top of timing, then stocks on top of both. Every layer must earn its place.
Report Sharpe / Sortino / maxDD / Calmar / turnover / exposure as diagnostics alongside.

---

# 9. EXPECTATION CALIBRATION — do NOT target-chase
Build the best HONEST model first, then SEE what it gives. "Keep adding/tuning until it
hits X%" is overfitting-by-iteration — the data does not care about a target. If the
honest robust answer is modest, report it as-is. Always report **return AND drawdown AND
turnover together**; a high line hiding a deep drawdown or cost-fragile turnover is a
failure, not a win. Remember the A-share market itself compounded only low-single-digits
over the last decade — beware any result that looks too good.

---

# 10. SUCCESS METRICS
Primary: Out-of-Sample CAGR and Out-of-Sample Calmar (within the §2 drawdown budget).
Secondary: terminal wealth, Sharpe (state if rf-adjusted), Sortino, annualized volatility,
maxDD, profit factor, win rate, avg holding period, turnover, exposure %, regime stability.
Preference: the SIMPLEST structure achieving robust OOS compounding within the drawdown
budget. Equal performance → prefer simpler, lower-turnover, less cost-sensitive designs.

---

# 11. PROCESS (report at each checkpoint: findings / assumptions / risks / next steps)
1. **Data audit** — coverage, anomalies, calendar gaps, delisting/ST/limit handling, PIT
   membership, ETF universe & liquidity-onset timeline, raw-vs-adjusted, units. Append a
   verification-ledger entry. Concise audit report.
2. **Infrastructure** — a FAST, vectorized, PARALLEL backtest framework (saturate all
   cores; never run a vectorizable computation as a single-core for-loop). Small, testable,
   independently runnable modules: data access, signal computation, portfolio construction
   (stocks + ETFs + cash, concentration knob), engine (honest costs/impact, limits,
   suspensions, delisting, ADV cap, ETF shallow-history), analytics, reporting.
3. **Layer research** — for EACH layer, design and COMPARE multiple methods head-to-head;
   let OOS compound return drive the choice; document failures too.
4. **Validation gauntlet** (§7) → one-shot terminal holdout. Final report: experiments
   incl. failures, OOS compound-return evidence vs CSI300 TR + naive top-down baseline,
   per-layer attribution, deflated-return accounting, drawdown / turnover / participation /
   regime breakdown, and a clear DECISION naming the single strategy to trust with real
   capital under THIS top-down, drawdown-controlled mandate.

---

# 12. CONTINUITY (sessions can be cut off at any time)
- As your FIRST actions, create `project_state.md` (current phase; completed / active /
  pending work; key discoveries; rejected ideas; assumptions; next actions) — a fresh
  session must resume solely from it. Keep it + a handoff summary CURRENT at all times.
- Commit completed work immediately. Run long backtests detached + resumable, writing to a
  LOCAL gitignored cache inside this folder. On any interrupt: save, commit, update state,
  write handoff — optimize for continuity, not completion.
- Deliverables in `reports/`: final report + equity/drawdown chart + benchmark table +
  reproducible code + an honest verdict on the achievable net return.

---

# 13. FINAL DECISION PRINCIPLE
Select the strategy with the highest robust OOS compound return that stays within the
drawdown budget (§2), performs consistently across regimes, beats CSI300 TR and the naive
top-down baseline on absolute return, AND whose complexity is justified by per-layer
attribution (every layer earns its place). Reject: overfit, cost-fragile, non-robust
designs, anything that cannot beat its benchmarks or explain where its return comes from.
The final strategy must be suitable for real capital deployment.

---

# APPENDIX A (OPTIONAL — delete for a fully blank slate)
Empirical priors from a SEPARATE prior study. Treat each as a HYPOTHESIS to confirm or
refute independently — do NOT take as given. Delete this appendix for zero influence.
- The market-timing layer is often the largest source of value; pure price-pattern stock
  selection may carry only a short-lived (~5–10 day) edge that decays, with a negative
  median forward return. Event-study every signal's edge horizon first.
- Expression matters: the same timing view on different broad indices gives very different
  risk/return; "broad-index ETF + timing" is a hard benchmark to beat (hence §8.2).
- Cross-sectional momentum/trend-following on individual A-shares may have NEGATIVE edge
  (cross-section can mean-revert) — do not import US-style momentum intuition; test it.
- Costs eat most of the thin edge of short-holding strategies — apply slippage/impact for real.
- A single recent holdout window can be regime-misleading (a lone bull window flatters
  buy-and-hold and penalizes timing) — span ≥ 2 regimes (hence §7).
