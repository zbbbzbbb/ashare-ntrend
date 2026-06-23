"""
Execution — trade execution simulation with cost model.

Handles entry/exit logic and integrates with the cost model.
All decisions are end-of-day only.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from .risk import RiskManager
from .position import PositionSizer


class ExecutionEngine:
    """End-of-day execution engine with full cost accounting."""
    
    def __init__(
        self,
        capital: float = 1_500_000.0,
        cost_model=None,
        max_weekly_buys: int = 2,
    ):
        self.capital = capital
        self.initial_capital = capital
        self.cost_model = cost_model
        self.max_weekly_buys = max_weekly_buys
        
        # State
        self.cash = capital
        self.positions = {}  # ts_code → {shares, avg_cost, entry_date, ...}
        self.risk_manager = RiskManager()
        self.sizer = PositionSizer(capital=capital)
        
        # Tracking
        self.trades = []  # List of all trades
        self.daily_nav = []  # [(date, nav)]
        self.weekly_buy_count = 0
        self.current_week = None
    
    def get_nav(self, current_prices: dict, current_date=None) -> float:
        """Calculate current NAV (cash + positions at market)."""
        position_value = 0.0
        for ts_code, pos in self.positions.items():
            price = current_prices.get(ts_code, pos["avg_cost"])
            position_value += price * pos["shares"]
        return self.cash + position_value
    
    def process_daily(
        self,
        date_str: str,
        current_prices: dict,  # {ts_code: close_price}
        bbi_data: dict,  # {ts_code: BBI DataFrame}
        bbi_idx_map: dict,  # {ts_code: int} — correct row index per stock
        can_enter: bool,
        buy_candidates: list,  # [(ts_code, score)]
    ) -> dict:
        """Process end-of-day for all positions and generate orders.
        
        Args:
            date_str: Current trade date (YYYYMMDD)
            current_prices: Latest close prices
            bbi_data: BBI signal DataFrames per stock
            bbi_idx_map: Correct row index into each BBI DataFrame for this date
            can_enter: Whether market timing allows new entries
            buy_candidates: Ranked list of (ts_code, score) for potential buys
        
        Returns:
            dict with summary of actions taken
        """
        week_id = date_str[:6]  # YYYYMM
        
        # Reset weekly counter if new week
        if week_id != self.current_week:
            self.weekly_buy_count = 0
            self.current_week = week_id
        
        actions = {"sells": [], "buys": [], "holds": []}
        
        # 1. Check exits for all positions
        to_close = []
        for ts_code, pos in list(self.positions.items()):
            if ts_code not in current_prices:
                continue  # Skip if we don't have current price data
            
            bbi_df = bbi_data.get(ts_code)
            if bbi_df is None:
                continue
            
            bbi_idx = bbi_idx_map.get(ts_code, -1)
            result = self.risk_manager.update(
                ts_code, current_prices[ts_code], bbi_df, bbi_idx
            )
            
            if result["action"] in ("stop_loss", "profit_take", "profit_take_close", "trend_exit"):
                shares_to_sell = result.get("shares_to_sell", pos["shares"])
                self._execute_sell(ts_code, current_prices[ts_code], shares_to_sell, 
                                  date_str, result["reason"])
                actions["sells"].append((ts_code, shares_to_sell, result["reason"]))
                
                if result["action"] in ("stop_loss", "profit_take_close", "trend_exit"):
                    to_close.append(ts_code)
                elif result["action"] == "profit_take":
                    # Partial sell — position remains
                    pos["shares"] -= shares_to_sell
        
        for ts_code in to_close:
            self.risk_manager.close_position(ts_code)
            self.positions.pop(ts_code, None)
        
        # 2. Check entries (only if market timing is bullish)
        if can_enter and self.weekly_buy_count < self.max_weekly_buys:
            for ts_code, score in buy_candidates:
                if ts_code in self.positions:
                    continue
                if self.weekly_buy_count >= self.max_weekly_buys:
                    break
                
                price = current_prices.get(ts_code)
                if price is None or price <= 0:
                    continue
                
                # Position size
                shares = self.sizer.size_position(
                    price=price,
                    current_positions=len(self.positions),
                )
                
                if shares > 0:
                    buy_bbi_idx = bbi_idx_map.get(ts_code, -1)
                    self._execute_buy(ts_code, price, shares, date_str, score,
                                     bbi_data.get(ts_code), buy_bbi_idx)
                    actions["buys"].append((ts_code, shares, score))
                    self.weekly_buy_count += 1
        
        # 3. Record daily NAV
        nav = self.get_nav(current_prices, date_str)
        self.daily_nav.append((date_str, nav))
        
        return actions
    
    def _execute_buy(self, ts_code: str, price: float, shares: int, 
                     date_str: str, score: float, bbi_df=None, bbi_idx: int = -1):
        """Execute a buy order with costs."""
        gross_cost = price * shares
        commission = self.cost_model.commission(gross_cost, is_etf=ts_code.startswith("5"))
        stamp_duty = 0  # No stamp duty on buys
        total_cost = gross_cost + commission
        
        if total_cost > self.cash:
            # Scale down
            affordable_shares = int((self.cash - commission) / price)
            affordable_shares = (affordable_shares // 100) * 100
            if affordable_shares <= 0:
                return
            shares = affordable_shares
            gross_cost = price * shares
            commission = self.cost_model.commission(gross_cost, is_etf=ts_code.startswith("5"))
            total_cost = gross_cost + commission
        
        self.cash -= total_cost
        
        # Determine BBI status at entry date
        above_bbi = False
        bbi_value = 0.0
        if bbi_df is not None and bbi_idx >= 0 and bbi_idx < len(bbi_df):
            above_bbi = bool(bbi_df["above_bbi"].iloc[bbi_idx])
            bbi_value = float(bbi_df["bbi"].iloc[bbi_idx])
        
        self.positions[ts_code] = {
            "shares": shares,
            "avg_cost": price,
            "entry_date": date_str,
        }
        
        self.risk_manager.open_position(
            ts_code, price, date_str, shares, above_bbi, bbi_value
        )
        
        self.trades.append({
            "date": date_str,
            "ts_code": ts_code,
            "action": "buy",
            "price": price,
            "shares": shares,
            "gross_cost": gross_cost,
            "commission": commission,
            "stamp_duty": 0,
            "net_cost": total_cost,
            "score": score,
        })
    
    def _execute_sell(self, ts_code: str, price: float, shares: int,
                      date_str: str, reason: str):
        """Execute a sell order with costs."""
        if ts_code not in self.positions:
            return
        
        pos = self.positions[ts_code]
        shares = min(shares, pos["shares"])
        
        gross_proceeds = price * shares
        commission = self.cost_model.commission(gross_proceeds, is_etf=ts_code.startswith("5"))
        
        is_etf = ts_code.startswith("5")
        stamp_duty = 0 if is_etf else self.cost_model.stamp_duty(gross_proceeds)
        
        net_proceeds = gross_proceeds - commission - stamp_duty
        self.cash += net_proceeds
        
        pnl = (price - pos["avg_cost"]) * shares - commission - stamp_duty
        pnl_pct = (price / pos["avg_cost"] - 1) * 100
        
        self.trades.append({
            "date": date_str,
            "ts_code": ts_code,
            "action": "sell",
            "price": price,
            "shares": shares,
            "gross_proceeds": gross_proceeds,
            "commission": commission,
            "stamp_duty": stamp_duty,
            "net_proceeds": net_proceeds,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": reason,
        })
    
    def get_performance_summary(self) -> dict:
        """Compute performance metrics."""
        if not self.daily_nav:
            return {}
        
        nav_df = pd.DataFrame(self.daily_nav, columns=["date", "nav"])
        nav_df["date"] = pd.to_datetime(nav_df["date"])
        nav_df = nav_df.set_index("date")
        
        nav_df["ret"] = nav_df["nav"].pct_change()
        
        # CAGR
        total_days = len(nav_df)
        years = total_days / 252
        total_return = nav_df["nav"].iloc[-1] / self.initial_capital - 1
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Max drawdown
        nav_df["peak"] = nav_df["nav"].cummax()
        nav_df["dd"] = (nav_df["nav"] - nav_df["peak"]) / nav_df["peak"]
        max_dd = nav_df["dd"].min()
        
        # Calmar ratio
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # Profit factor
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            sells = trades_df[trades_df["action"] == "sell"]
            gross_profits = sells[sells["pnl"] > 0]["pnl"].sum()
            gross_losses = abs(sells[sells["pnl"] < 0]["pnl"].sum())
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else float("inf")
            
            win_rate = len(sells[sells["pnl"] > 0]) / len(sells) if len(sells) > 0 else 0
            avg_holding = None  # Would need entry/exit matching
        else:
            profit_factor = 0
            win_rate = 0
        
        return {
            "initial_capital": self.initial_capital,
            "final_nav": nav_df["nav"].iloc[-1],
            "total_return_pct": total_return * 100,
            "cagr_pct": cagr * 100,
            "max_drawdown_pct": max_dd * 100,
            "calmar_ratio": calmar,
            "profit_factor": profit_factor,
            "win_rate_pct": win_rate * 100,
            "total_trades": len(self.trades),
            "num_buys": len([t for t in self.trades if t["action"] == "buy"]),
            "num_sells": len([t for t in self.trades if t["action"] == "sell"]),
            "years": round(years, 2),
        }
