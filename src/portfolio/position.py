"""
Position Sizing — determines trade size for 1.5M RMB account.

Constraints:
  - Max positions: ~10 stocks
  - Max single position: ~15% of capital
  - Respect liquidity (daily turnover-based limits)
  - Account for trading costs
"""

import numpy as np


class PositionSizer:
    """Position sizing for 1.5M RMB long-only account."""
    
    def __init__(
        self,
        capital: float = 1_500_000.0,
        max_positions: int = 10,
        max_position_pct: float = 0.15,
        max_sector_pct: float = 0.40,
        liquidity_pct: float = 0.01,  # Max % of daily volume
    ):
        self.capital = capital
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.liquidity_pct = liquidity_pct
    
    def size_position(
        self,
        price: float,
        daily_volume_shares: float = None,
        daily_turnover_yuan: float = None,
        current_positions: int = 0,
        current_industry_exposure: float = 0.0,
    ) -> int:
        """Calculate shares for a new position.
        
        Args:
            price: Current stock price
            daily_volume_shares: Average daily volume in shares (for liquidity check)
            daily_turnover_yuan: Average daily turnover in yuan
            current_positions: Number of existing positions
            current_industry_exposure: Current exposure to this industry (yuan)
        
        Returns:
            Number of shares to buy (0 if constrained out)
        """
        remaining_slots = self.max_positions - current_positions
        if remaining_slots <= 0:
            return 0
        
        # Equal-weight target for remaining slots
        target_position_value = self.capital * self.max_position_pct
        
        # Can't exceed max industry exposure
        max_additional_industry = self.capital * self.max_sector_pct - current_industry_exposure
        if max_additional_industry <= 0:
            return 0
        target_position_value = min(target_position_value, max_additional_industry)
        
        # Liquidity constraint: don't exceed 1% of daily volume
        if daily_volume_shares and daily_volume_shares > 0:
            max_liquidity_shares = int(daily_volume_shares * self.liquidity_pct)
        else:
            max_liquidity_shares = float("inf")
        
        if daily_turnover_yuan and daily_turnover_yuan > 0:
            max_liquidity_value = daily_turnover_yuan * self.liquidity_pct * 5  # 5% of daily turnover
        else:
            max_liquidity_value = float("inf")
        
        # Calculate shares
        target_shares = int(target_position_value / price)
        target_shares = min(target_shares, max_liquidity_shares)
        
        # Ensure position value doesn't exceed liquidity value cap
        if target_shares * price > max_liquidity_value:
            target_shares = int(max_liquidity_value / price)
        
        # Minimum lot: 100 shares for A-shares
        target_shares = (target_shares // 100) * 100
        
        return max(0, target_shares)
    
    def rebalance_check(self, current_prices: dict, positions: dict) -> list:
        """Check if any positions need rebalancing.
        
        Returns list of (ts_code, action, reason).
        """
        # For trend-following, we don't typically rebalance —
        # positions are managed by stop-loss and profit-taking rules.
        return []
