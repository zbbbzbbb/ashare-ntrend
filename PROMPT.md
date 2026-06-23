# A-share Trend Following Strategy — Research Brief

(Standalone brief for a NEW, independent strategy project. Execute in a fresh session.)

---

# 0. OBJECTIVE

You are an experienced quantitative researcher and engineer.

Build and validate a long-only A-share trend-following strategy whose objective is:

**Maximize robust out-of-sample long-term compound return through trend participation while maintaining disciplined risk management.**

The strategy is based on:

1. Market Timing
2. Industry Rotation
3. N-Pattern Trend Breakouts
4. Trend-Following Position Management

The strategy assumes that excess return in A-shares primarily comes from:

* Correct market regime identification
* Industry leadership rotation
* Trend persistence
* Capital concentration in leading sectors

Return should NOT primarily rely on:

* Accounting factors
* Financial statement quality
* Statistical arbitrage
* Mean reversion
* Black-box machine learning

Capital preservation should be achieved through:

* Market timing
* Trend exits
* Position management

rather than excessive diversification.

---

# 1. HARD CONSTRAINTS

## Independence

Derive all research independently.

Do not reuse strategy logic from any other project.

Only shared market data may be used.

---

## Long Only

Allowed assets:

* A-share stocks
* Listed ETFs
* Cash

Not allowed:

* Shorting
* Leverage
* Margin
* Futures
* Options
* Leveraged ETFs

---

## Permitted Data

Allowed:

* Adjusted price
* Volume
* Turnover
* Volatility
* Market cap
* PE
* PB
* Dividend yield
* ETF data
* Index data
* Market breadth
* Northbound holdings
* Margin financing
* Macro data

---

# 2. CORE STRATEGY FRAMEWORK

Research must focus exclusively on this framework.

No unrelated factor strategies.

No low-volatility strategy.

No mean-reversion strategy.

No alternative alpha engines.

---

## STEP 1 — MARKET TIMING

Primary benchmark:

CSI300

Bullish Regime:

1. DIF > DEA

2. MACD Histogram Expanding

3. DIF Rising

Highest Priority:

DIF moving from below zero toward zero

("0下到0上")

Only bullish regimes allow new positions.

Bearish Regime:

1. DIF < DEA

Actions:

* No new positions
* Existing positions may be reduced or exited
* Daily process becomes Sell Only

Research Questions:

* Best MACD regime definition
* Best index choice
* Regime robustness across cycles

---

## STEP 2 — INDUSTRY ROTATION

Universe:

三级行业指数

88-series industry indices

Industry Selection Requirements:

1. MACD Bullish

DIF > DEA

2. Relative Strength Leadership

Top-ranked industries over 20 trading days

3. Volume Expansion

Industry turnover above 60-day average

4. N-Pattern Structure

Definition:

First Leg:

20 days ago → 10 days ago rising

Pullback:

10 days ago → 5 days ago retracing

Second Leg:

Current price breaks recent high

Priority:

Industries attracting incremental capital.

Research Questions:

* Relative strength ranking methods
* Capital inflow proxies
* Industry concentration levels

---

## STEP 3 — STOCK SELECTION

Universe:

Constituents of selected industries only.

Requirements:

1. MACD Bullish

DIF > DEA

2. N-Pattern Breakout

3. Volume Confirmation

Volume > 20-day average volume

4. Relative Strength

Above industry median

5. Above BBI preferred

Exclude:

ST

*ST

Listed less than 250 trading days

Delisting risk

Extremely illiquid stocks

Research Questions:

* Best breakout confirmation
* Best volume confirmation
* Best stock ranking model

---

## STEP 4 — ENTRY

Preferred Entry:

N-pattern breakout.

Highest Conviction Setup:

MACD transitioning from below zero toward zero.

Research:

* Breakout confirmation methods
* Pullback depth requirements
* Entry timing robustness

---

## STEP 5 — RISK MANAGEMENT

Case A:

Entry below BBI.

Exit:

Price falls 3% below entry cost.

Case B:

Entry above BBI.

Exit:

Two consecutive daily closes below BBI.

All stop-loss decisions occur only during end-of-day review.

Research Questions:

* BBI effectiveness
* Alternative trend exits
* Stop-loss robustness

---

## STEP 6 — PROFIT TAKING

Default Rule:

When portfolio gain reaches approximately +20%:

Sell 50%.

Hold remaining 50% until trend exit.

Alternative Variants:

Variant A:

20% gain → reduce 50%

Variant B:

20% gain → reduce 30%

35% gain → reduce additional 30%

Remainder follows trend

Research must compare variants.

---

## STEP 7 — TRADING FREQUENCY

Maximum:

2 discretionary buy transactions per week.

If market timing is bearish:

No new positions.

Daily Process:

End-of-day review only.

Stop-loss execution occurs only after close.

Avoid unnecessary turnover.

---

# 3. CAPITAL AND EXECUTION

Capital:

1,500,000 RMB

All strategies must be executable at this size.

Respect:

* Liquidity constraints
* Participation limits
* Trading costs
* Slippage

---

# 4. COST MODEL

Mandatory:

Stocks:

* Commission
* Stamp duty

ETFs:

* Commission

Additional:

* Slippage
* Market impact

Run:

Normal Cost

and

2× Cost Stress Test

No result is valid without full costs.

---

# 5. DATA

Data source:

Tushare Data Lake

Default location:

~/quant-data

Structure:

lake/
catalog.duckdb
_manifests/

History:

1990-12-19
to latest available trade date

Research must verify:

- PIT correctness
- Delisting handling
- ST handling
- Suspension handling
- ETF history quality
- Corporate action adjustments

All signals and returns must use adjusted prices.

Available datasets include:

Stocks:
- daily
- adj_factor
- daily_basic
- stock_basic
- hk_hold
- margin

Funds:
- fund_daily
- fund_adj
- fund_basic

Indices:
- index_daily
- index_member_all
- index_weight
- index_classify

Rates:
- shibor
- yc_cb

Reference:
- trade_cal

---

# 6. RISK PREFERENCE

This strategy seeks higher compound return through trend following.

Large drawdowns should be reduced through timing and exits.

Target:

Max Drawdown <= 35%

Preferred:

Max Drawdown <= 30%

Strategies requiring persistent drawdowns greater than 50% are not preferred.

---

# 7. VALIDATION

Strict Anti-Overfitting Requirements.

Mandatory:

1. Walk-Forward Validation

2. CPCV

3. Purged Cross Validation

4. Embargo

5. Parameter Stability

6. Trial Count Tracking

7. Deflated Sharpe Analysis

8. Regime Breakdown

9. Final Holdout Evaluation

No single in-sample backtest is acceptable.

---

# 8. RESEARCH PRIORITIES

Highest Priority Research Question:

Does N-pattern trend continuation generate robust out-of-sample alpha?

Research Areas:

1. N-pattern definitions

2. Pullback depth

3. Breakout confirmation

4. MACD regime filters

5. BBI exits

6. Industry rotation effectiveness

7. Market timing effectiveness

Focus on robustness rather than optimization.

---

# 9. SUCCESS METRICS

Primary:

Out-of-Sample CAGR

Secondary:

Terminal Wealth

Calmar Ratio

Maximum Drawdown

Profit Factor

Win Rate

Average Holding Period

Turnover

Regime Stability

The preferred strategy is the one that achieves the highest robust out-of-sample compound return without relying on excessive drawdowns.

---

# 10. PROCESS

1. Data Audit

2. Infrastructure

3. Signal Research

4. Market Timing Research

5. Industry Rotation Research

6. Stock Selection Research

7. Portfolio Construction

8. Validation

9. Holdout Testing

10. Final Decision

Every stage must document:

* Findings
* Assumptions
* Risks
* Next Actions

---

# 11. FINAL DECISION PRINCIPLE

Select the strategy that demonstrates:

* Highest robust out-of-sample CAGR
* Consistent trend capture
* Acceptable drawdown
* Stable performance across regimes

Reject:

* Overfit strategies
* Cost-fragile strategies
* Non-robust strategies

The final strategy should be suitable for real capital deployment under a long-term trend-following framework.
