"""
Backtest Engine — event-driven daily loop integrating all strategy modules.

Uses quant-data DuckDB catalog. Computes synthetic industry returns from 
constituent stocks since industry index daily prices aren't in the catalog.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import (
    get_connection, get_trade_dates, get_index_daily,
    get_all_l3_index_codes, get_industry_stocks, get_l3_to_stocks_map,
    get_daily_data, get_stock_universe,
)
from signals.macd import compute_macd
from signals.bbi import compute_bbi_signal
from timing.market_timing import MarketTiming
from rotation.industry import IndustryRotator
from selection.stock_select import StockSelector
from portfolio.risk import RiskManager
from portfolio.position import PositionSizer
from portfolio.execution import ExecutionEngine
from backtest.costs import CostModel


class BacktestEngine:
    """Main backtest orchestrator."""
    
    def __init__(
        self,
        start_date: str = "20150101",
        end_date: str = "20250101",
        capital: float = 1_500_000.0,
        benchmark_code: str = "000300.SH",
        cost_stress: float = 1.0,
        top_industries: int = 5,
        max_positions: int = 10,
        max_weekly_buys: int = 2,
        profit_variant: str = "default",
        macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
        n_leg1_start: int = 20, n_leg1_end: int = 10, n_pullback_end: int = 5,
        verbose: bool = True,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.capital = capital
        self.benchmark_code = benchmark_code
        self.top_industries = top_industries
        self.max_positions = max_positions
        self.max_weekly_buys = max_weekly_buys
        self.profit_variant = profit_variant
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.n_leg1_start = n_leg1_start
        self.n_leg1_end = n_leg1_end
        self.n_pullback_end = n_pullback_end
        self.verbose = verbose
        
        self.cost_model = CostModel(stress_multiplier=cost_stress)
        self.market_timing = MarketTiming(benchmark_code, macd_fast, macd_slow, macd_signal)
        self.rotator = IndustryRotator(
            top_n=top_industries, macd_fast=macd_fast, macd_slow=macd_slow, 
            macd_signal=macd_signal, n_leg1_start=n_leg1_start,
            n_leg1_end=n_leg1_end, n_pullback_end=n_pullback_end,
        )
        self.selector = StockSelector(
            macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal,
            n_leg1_start=n_leg1_start, n_leg1_end=n_leg1_end, n_pullback_end=n_pullback_end,
            max_positions=max_positions,
        )
        self.execution = ExecutionEngine(
            capital=capital, cost_model=self.cost_model, max_weekly_buys=max_weekly_buys,
        )
        self.results = None
    
    def run(self) -> dict:
        """Execute the full backtest."""
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"BACKTEST: {self.start_date} → {self.end_date}")
            print(f"Capital: {self.capital:,.0f} RMB | Cost stress: {self.cost_model.stress_multiplier}×")
            print(f"Top industries: {self.top_industries} | Max positions: {self.max_positions}")
            print(f"{'='*60}")
        
        # 1. Trading calendar
        trade_dates = get_trade_dates(self.start_date, self.end_date)
        if len(trade_dates) == 0:
            print("ERROR: No trading dates found!")
            return {}
        dates = trade_dates["cal_date"].tolist()
        
        if self.verbose:
            print(f"Trading days: {len(dates):,}")
        
        # 2. Benchmark for market timing
        benchmark_df = get_index_daily(self.benchmark_code, self.start_date, self.end_date)
        if len(benchmark_df) == 0:
            print(f"ERROR: No benchmark data for {self.benchmark_code}!")
            return {}
        benchmark_df = benchmark_df.sort_values("trade_date").reset_index(drop=True)
        
        # 3. Fit market timing
        if self.verbose:
            print("Fitting market timing...")
        self.market_timing.fit(benchmark_df["close"])
        regime_stats = self.market_timing.get_regime_stats()
        if self.verbose:
            print(f"  Bullish: {regime_stats.get('bullish_pct', 0):.1f}% | "
                  f"Bearish: {regime_stats.get('bearish_pct', 0):.1f}%")
        
        # 4. Pre-load industry → stock mapping
        if self.verbose:
            print("Loading industry→stock mappings...")
        l3_codes = get_all_l3_index_codes()
        if self.verbose:
            print(f"  L3 industries: {len(l3_codes)}")
        
        mid_date = dates[len(dates) // 2]
        l3_to_stocks = get_l3_to_stocks_map(mid_date, l3_codes)
        if self.verbose:
            print(f"  Industries with constituents: {len(l3_to_stocks)}")
        
        # 5. Pre-load broad stock universe data for the full period
        if self.verbose:
            print("Pre-loading broad stock universe...")
        
        # Load top stocks by market cap at midpoint
        # circ_mv is in 万元; 200000 = ~2B yuan
        ref_universe = get_stock_universe(mid_date, exclude_st=True, 
                                          min_listed_days=250, min_market_cap=200000)
        broad_codes = ref_universe["ts_code"].tolist()[:1000]
        if self.verbose:
            print(f"  Broad universe: {len(broad_codes)} stocks")
        
        # Load all data from actual start date for full coverage
        stock_cache = self._load_stock_batch(broad_codes, self.start_date, self.end_date, {})
        if self.verbose:
            print(f"  Loaded {len(stock_cache)} stocks from {self.start_date}")
        
        # Pre-compute industry composites ONCE
        if self.verbose:
            print("Building industry composites...")
        industry_composites = self._build_industry_composites(
            stock_cache, l3_to_stocks, self.start_date, self.end_date
        )
        if self.verbose:
            print(f"  Built {len(industry_composites)} industry composites")
        
        # 6. Main loop
        warmup = 60
        if len(dates) <= warmup:
            print("ERROR: Not enough data for warmup")
            return {}
        
        bbi_cache = {}
        
        if self.verbose:
            print(f"Running daily loop ({len(dates)} days)...")
        
        for i, date_str in enumerate(dates):
            if i < warmup:
                continue
            
            if self.verbose and i % 200 == 0:
                nav = self.execution.get_nav({})
                print(f"  {date_str} (day {i}/{len(dates)}) | NAV: {nav:,.0f} | "
                      f"Positions: {len(self.execution.positions)}")
            
            # Benchmark index
            bench_rows = benchmark_df[benchmark_df["trade_date"] == date_str]
            if len(bench_rows) == 0:
                continue
            bench_idx = bench_rows.index[0]
            can_enter = self.market_timing.can_enter(bench_idx)
            
            buy_candidates = []
            
            # ── Industry Rotation + Stock Selection ──
            if can_enter and self.execution.weekly_buy_count < self.max_weekly_buys:
                # Slice industry composites up to current date
                ind_data_sliced = {}
                for l3_code, df in industry_composites.items():
                    sliced = df[df["trade_date"] <= date_str]
                    if len(sliced) >= 60:
                        ind_data_sliced[l3_code] = sliced.reset_index(drop=True)
                
                if len(ind_data_sliced) > 0:
                    scored_ind = self.rotator.score_industries(ind_data_sliced)
                    
                    if len(scored_ind) > 0:
                        qualified_ind = scored_ind[scored_ind["macd_bullish"] > 0]
                        if len(qualified_ind) < self.top_industries:
                            qualified_ind = scored_ind.head(self.top_industries)
                        
                        selected_l3 = qualified_ind.head(self.top_industries)["index_code"].tolist()
                        
                        # Get candidate stocks from selected industries
                        candidate_codes = set()
                        for l3 in selected_l3:
                            candidate_codes.update(l3_to_stocks.get(l3, []))
                        
                        candidate_codes = list(candidate_codes & set(stock_cache.keys()))
                        
                        if candidate_codes:
                            # Slice each stock's data to include enough history
                            candidate_data = {}
                            for c in candidate_codes:
                                df = stock_cache[c]
                                sliced = df[df["trade_date"] <= date_str]
                                if len(sliced) >= 60:
                                    candidate_data[c] = sliced.reset_index(drop=True)
                            
                            if candidate_data:
                                screened = self.selector.screen_stocks(candidate_data, -1)
                                
                                if len(screened) > 0:
                                    qualified = screened[
                                        (screened["macd_bullish"] > 0) &
                                        (screened["volume_ok"] > 0)
                                    ]
                                    if len(qualified) > 0:
                                        qualified = qualified.sort_values("score", ascending=False)
                                        buy_candidates = [
                                            (row["ts_code"], row["score"])
                                            for _, row in qualified.head(10).iterrows()
                                        ]
            
            # ── Collect current prices ──
            # Ensure position stocks have data (load on demand if missing)
            pos_codes = list(self.execution.positions.keys())
            missing_pos = [c for c in pos_codes if c not in stock_cache]
            if missing_pos:
                # Load recent data for missing position stocks
                lookback = dates[max(0, i - 200)]
                stock_cache = self._load_stock_batch(missing_pos, lookback, date_str, stock_cache)
            
            current_prices = {}
            needed = pos_codes + [c[0] for c in buy_candidates]
            for ts_code in needed:
                if ts_code in stock_cache:
                    df = stock_cache[ts_code]
                    date_rows = df[df["trade_date"] == date_str]
                    if len(date_rows) > 0:
                        current_prices[ts_code] = float(date_rows["close_adj"].iloc[-1])
            
            # ── Build BBI data + correct row indices ──
            bbi_data = {}
            bbi_idx_map = {}
            for ts_code in self.execution.positions.keys():
                if ts_code in stock_cache and ts_code not in bbi_cache:
                    df = stock_cache[ts_code]
                    bbi_cache[ts_code] = compute_bbi_signal(df["close_adj"])
                if ts_code in bbi_cache and ts_code in stock_cache:
                    bbi_data[ts_code] = bbi_cache[ts_code]
                    # Find the row index in stock_cache for this date
                    stock_df = stock_cache[ts_code]
                    matches = stock_df[stock_df["trade_date"] == date_str]
                    if len(matches) > 0:
                        bbi_idx_map[ts_code] = int(matches.index[0])
            
            # Also compute for buy candidates (needed for _execute_buy BBI check)
            for ts_code, _ in buy_candidates:
                if ts_code not in bbi_idx_map and ts_code in stock_cache:
                    if ts_code not in bbi_cache:
                        df = stock_cache[ts_code]
                        bbi_cache[ts_code] = compute_bbi_signal(df["close_adj"])
                    # Put BBI data where _execute_buy can find it
                    bbi_data[ts_code] = bbi_cache[ts_code]
                    matches = stock_cache[ts_code]
                    matches = matches[matches["trade_date"] == date_str]
                    if len(matches) > 0:
                        bbi_idx_map[ts_code] = int(matches.index[0])
            
            # ── Process daily ──
            self.execution.process_daily(
                date_str, current_prices, bbi_data, bbi_idx_map,
                can_enter, buy_candidates
            )
        
        # ── Results ──
        self.results = self.execution.get_performance_summary()
        
        if self.verbose:
            r = self.results
            print(f"\n{'='*60}")
            print("BACKTEST COMPLETE")
            print(f"Final NAV: {r.get('final_nav', 0):,.0f}")
            print(f"Total Return: {r.get('total_return_pct', 0):.2f}%")
            print(f"CAGR: {r.get('cagr_pct', 0):.2f}%")
            print(f"Max DD: {r.get('max_drawdown_pct', 0):.2f}%")
            print(f"Calmar: {r.get('calmar_ratio', 0):.2f}")
            print(f"Profit Factor: {r.get('profit_factor', 0):.2f}")
            print(f"Win Rate: {r.get('win_rate_pct', 0):.1f}%")
            print(f"Trades: {r.get('total_trades', 0)}")
            print(f"{'='*60}")
        
        return self.results
    
    def _build_industry_composites(self, stock_cache: dict, 
                                    l3_to_stocks: dict,
                                    start: str, end: str) -> dict:
        """Build synthetic industry price series from constituent stocks.
        
        Computes equal-weighted composite close_adj for each industry
        that has at least 5 constituent stocks with data.
        
        Returns: {l3_code: DataFrame with trade_date, close, vol, ...}
        """
        industry_data = {}
        
        for l3_code, stocks in l3_to_stocks.items():
            available = [s for s in stocks if s in stock_cache]
            if len(available) < 3:  # Lower threshold to get more industries
                continue
            
            # Collect all dates across constituents
            all_dates = set()
            for s in available:
                all_dates.update(stock_cache[s]["trade_date"].tolist())
            
            if not all_dates:
                continue
            
            # Build composite: equal-weighted average of close_adj
            date_list = sorted(all_dates)
            composites = []
            
            for d in date_list:
                closes = []
                vols = []
                for s in available:
                    df = stock_cache[s]
                    rows = df[df["trade_date"] == d]
                    if len(rows) > 0:
                        closes.append(float(rows["close_adj"].iloc[0]))
                        vols.append(float(rows["vol"].iloc[0]))
                
                if len(closes) >= 2:  # Need at least 2 stocks for valid composite
                    composites.append({
                        "trade_date": d,
                        "close": np.mean(closes),
                        "vol": np.sum(vols),
                        "open": np.mean(closes),  # Approximation
                        "high": np.mean(closes) * 1.01,
                        "low": np.mean(closes) * 0.99,
                    })
            
            if len(composites) >= 30:
                industry_data[l3_code] = pd.DataFrame(composites)
        
        return industry_data
    
    def _load_stock_batch(self, ts_codes: List[str], start: str, end: str,
                          cache: dict) -> dict:
        """Load adjusted stock data, updating cache."""
        new_codes = [c for c in ts_codes if c not in cache]
        
        if new_codes:
            batch_size = 300
            for j in range(0, len(new_codes), batch_size):
                batch = new_codes[j:j+batch_size]
                try:
                    df = get_daily_data(batch, start, end)
                    if len(df) > 0:
                        for ts_code, group in df.groupby("ts_code"):
                            group = group.sort_values("trade_date").reset_index(drop=True)
                            cache[ts_code] = group
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: stock load error: {e}")
        
        return cache


def run_backtest(params: dict = None) -> dict:
    """Convenience function to run a backtest."""
    if params is None:
        params = {}
    engine = BacktestEngine(**params)
    return engine.run()
