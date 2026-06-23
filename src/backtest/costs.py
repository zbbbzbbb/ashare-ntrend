"""
Cost Model — realistic A-share trading costs.

Stocks:
  - Commission: 0.025% (2.5‱) per trade, min 5 RMB
  - Stamp Duty: 0.05% (5‱) on sells only
  - Slippage: 0.1% (1‱) per trade
  - Market Impact: 0.05% (0.5‱) per trade

ETFs:
  - Commission: 0.025% per trade, min 5 RMB (may be lower in practice)
  - No stamp duty
  - Slippage: 0.05%

Stress test: 2× costs.
"""


class CostModel:
    """A-share cost model with normal and stress modes."""
    
    def __init__(
        self,
        commission_rate: float = 0.00025,   # 2.5‱
        commission_min: float = 5.0,          # 5 RMB minimum
        stamp_duty_rate: float = 0.0005,      # 5‱ (sells only)
        slippage_rate: float = 0.001,         # 10‱
        market_impact_rate: float = 0.0005,   # 5‱
        etf_commission_rate: float = 0.00025,
        etf_slippage_rate: float = 0.0005,
        stress_multiplier: float = 1.0,
    ):
        self.commission_rate = commission_rate * stress_multiplier
        self.commission_min = commission_min
        self.stamp_duty_rate = stamp_duty_rate * stress_multiplier
        self.slippage_rate = slippage_rate * stress_multiplier
        self.market_impact_rate = market_impact_rate * stress_multiplier
        self.etf_commission_rate = etf_commission_rate * stress_multiplier
        self.etf_slippage_rate = etf_slippage_rate * stress_multiplier
        self.stress_multiplier = stress_multiplier
    
    def commission(self, trade_value: float, is_etf: bool = False) -> float:
        """Calculate commission for a trade.
        
        Args:
            trade_value: Gross trade value (price × shares)
            is_etf: True if ETF
        
        Returns:
            Commission in RMB
        """
        if is_etf:
            rate = self.etf_commission_rate
        else:
            rate = self.commission_rate
        
        commission = max(trade_value * rate, self.commission_min)
        return round(commission, 2)
    
    def stamp_duty(self, trade_value: float) -> float:
        """Calculate stamp duty (sells only, stocks only).
        
        Args:
            trade_value: Gross trade value
        
        Returns:
            Stamp duty in RMB
        """
        return round(trade_value * self.stamp_duty_rate, 2)
    
    def slippage(self, trade_value: float, is_etf: bool = False) -> float:
        """Estimate slippage cost.
        
        Args:
            trade_value: Gross trade value
            is_etf: True if ETF
        
        Returns:
            Slippage in RMB
        """
        rate = self.etf_slippage_rate if is_etf else self.slippage_rate
        return round(trade_value * rate, 2)
    
    def market_impact(self, trade_value: float, daily_turnover: float = None) -> float:
        """Estimate market impact cost.
        
        For small retail trades (1.5M account), impact is minimal.
        Scales with trade_value / daily_turnover if provided.
        
        Args:
            trade_value: Gross trade value
            daily_turnover: Stock's average daily turnover (optional)
        
        Returns:
            Market impact in RMB
        """
        if daily_turnover and daily_turnover > 0:
            participation = trade_value / daily_turnover
            impact_rate = self.market_impact_rate * (1 + participation * 10)
        else:
            impact_rate = self.market_impact_rate
        
        return round(trade_value * impact_rate, 2)
    
    def total_buy_cost(self, price: float, shares: int, is_etf: bool = False,
                       daily_turnover: float = None) -> dict:
        """Calculate total cost for a buy order.
        
        Returns dict with cost breakdown.
        """
        gross = price * shares
        comm = self.commission(gross, is_etf)
        slip = self.slippage(gross, is_etf)
        impact = self.market_impact(gross, daily_turnover)
        
        return {
            "gross_value": gross,
            "commission": comm,
            "stamp_duty": 0.0,  # No stamp duty on buys
            "slippage": slip,
            "market_impact": impact,
            "total_cost": gross + comm + slip + impact,
        }
    
    def total_sell_cost(self, price: float, shares: int, is_etf: bool = False,
                        daily_turnover: float = None) -> dict:
        """Calculate total cost for a sell order."""
        gross = price * shares
        comm = self.commission(gross, is_etf)
        stamp = 0.0 if is_etf else self.stamp_duty(gross)
        slip = self.slippage(gross, is_etf)
        impact = self.market_impact(gross, daily_turnover)
        
        return {
            "gross_value": gross,
            "commission": comm,
            "stamp_duty": stamp,
            "slippage": slip,
            "market_impact": impact,
            "total_deductions": comm + stamp + slip + impact,
            "net_proceeds": gross - comm - stamp - slip - impact,
        }
    
    def get_effective_buy_price(self, price: float, is_etf: bool = False) -> float:
        """Effective buy price including costs per share (for PnL calc)."""
        # Simplified: markup by commission + slippage
        markup = self.commission_rate + self.slippage_rate
        if not is_etf:
            markup += self.market_impact_rate
        return price * (1 + markup)
    
    def get_effective_sell_price(self, price: float, is_etf: bool = False) -> float:
        """Effective sell price after costs per share (for PnL calc)."""
        deduction = self.commission_rate + self.slippage_rate + self.market_impact_rate
        if not is_etf:
            deduction += self.stamp_duty_rate
        return price * (1 - deduction)


def get_normal_cost() -> CostModel:
    """Get standard cost model."""
    return CostModel(stress_multiplier=1.0)


def get_stress_cost() -> CostModel:
    """Get 2× stress cost model."""
    return CostModel(stress_multiplier=2.0)
