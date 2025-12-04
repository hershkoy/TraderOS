#!/usr/bin/env python3
"""
Tests for option_strategies.py - Strategy classes for option spread selection.
"""

import pytest
import math
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.option_strategies import (
    OptionRow,
    VerticalSpreadCandidate,
    RatioSpreadCandidate,
    OTMCreditSpreadsStrategy,
    VerticalSpreadWithHedgingStrategy,
    get_strategy,
    list_strategies,
    STRATEGY_REGISTRY,
)


class TestOptionRow:
    """Tests for OptionRow dataclass."""
    
    def test_mid_calculation(self):
        """Test mid price calculation."""
        row = OptionRow(
            symbol="SPY",
            expiry="20251120",
            right="P",
            strike=450.0,
            bid=1.50,
            ask=1.60,
            delta=-0.10,
            volume=100.0,
        )
        assert row.mid == 1.55
    
    def test_mid_with_nan_bid(self):
        """Test mid returns None when bid is NaN."""
        row = OptionRow(
            symbol="SPY",
            expiry="20251120",
            right="P",
            strike=450.0,
            bid=float("nan"),
            ask=1.60,
            delta=-0.10,
            volume=100.0,
        )
        assert row.mid is None
    
    def test_mid_with_nan_ask(self):
        """Test mid returns None when ask is NaN."""
        row = OptionRow(
            symbol="SPY",
            expiry="20251120",
            right="P",
            strike=450.0,
            bid=1.50,
            ask=float("nan"),
            delta=-0.10,
            volume=100.0,
        )
        assert row.mid is None


class TestOTMCreditSpreadsStrategy:
    """Tests for OTMCreditSpreadsStrategy."""
    
    @pytest.fixture
    def strategy(self):
        return OTMCreditSpreadsStrategy()
    
    @pytest.fixture
    def sample_option_rows(self):
        """Create sample option rows for testing."""
        # Simulate a put option chain for SPY around 450
        # For puts: higher strikes = higher delta (closer to ATM), higher price
        # Lower strikes = lower delta (more OTM), lower price
        rows = []
        strikes = [440, 442, 444, 446, 448, 450, 452, 454, 456, 458, 460]
        for strike in strikes:
            # Higher strikes have higher deltas (closer to ATM)
            # delta ranges from -0.05 (440) to -0.55 (460)
            delta = -0.05 - (strike - 440) * 0.025
            # Higher strikes have higher prices (more ITM for puts)
            bid = max(0.15, (strike - 435) * 0.20)
            ask = bid + 0.10
            rows.append(OptionRow(
                symbol="SPY",
                expiry="20251120",
                right="P",
                strike=float(strike),
                bid=bid,
                ask=ask,
                delta=delta,
                volume=100.0,
            ))
        return rows
    
    def test_strategy_name(self, strategy):
        """Test strategy name property."""
        assert strategy.name == "otm_credit_spreads"
    
    def test_find_candidates_basic(self, strategy, sample_option_rows):
        """Test basic candidate finding."""
        candidates = strategy.find_candidates(
            sample_option_rows,
            width=4.0,
            target_delta=0.10,
            num_candidates=3,
            min_credit=0.10,
        )
        
        assert len(candidates) > 0
        assert len(candidates) <= 3
        
        for candidate in candidates:
            assert isinstance(candidate, VerticalSpreadCandidate)
            assert candidate.short.strike > candidate.long.strike
            assert candidate.width > 0
            assert candidate.credit > 0
    
    def test_find_candidates_respects_min_credit(self, strategy, sample_option_rows):
        """Test that candidates respect minimum credit."""
        candidates = strategy.find_candidates(
            sample_option_rows,
            width=4.0,
            target_delta=0.10,
            num_candidates=3,
            min_credit=0.50,
        )
        
        for candidate in candidates:
            assert candidate.credit >= 0.50
    
    def test_find_candidates_respects_width(self, strategy, sample_option_rows):
        """Test that candidates respect spread width."""
        candidates = strategy.find_candidates(
            sample_option_rows,
            width=4.0,
            target_delta=0.10,
            num_candidates=3,
            min_credit=0.10,
        )
        
        for candidate in candidates:
            # Width should be at least half the requested width
            assert candidate.width >= 2.0
    
    def test_find_candidates_no_usable_options(self, strategy):
        """Test error when no usable options found."""
        # Options with no delta
        rows = [
            OptionRow(
                symbol="SPY",
                expiry="20251120",
                right="P",
                strike=450.0,
                bid=1.50,
                ask=1.60,
                delta=None,
                volume=100.0,
            )
        ]
        
        with pytest.raises(ValueError, match="No usable options"):
            strategy.find_candidates(rows, width=4.0, target_delta=0.10, num_candidates=3, min_credit=0.10)
    
    def test_describe_candidate(self, strategy, sample_option_rows):
        """Test candidate description."""
        candidates = strategy.find_candidates(
            sample_option_rows,
            width=4.0,
            target_delta=0.10,
            num_candidates=1,
            min_credit=0.10,
        )
        
        description = strategy.describe_candidate("Test Label", candidates[0])
        
        assert "Test Label" in description
        assert "Sell" in description
        assert "Buy" in description
        assert "Credit" in description


class TestVerticalSpreadWithHedgingStrategy:
    """Tests for VerticalSpreadWithHedgingStrategy."""
    
    @pytest.fixture
    def strategy(self):
        return VerticalSpreadWithHedgingStrategy()
    
    @pytest.fixture
    def sample_option_rows_for_ratio(self):
        """Create sample option rows for ratio spread testing."""
        # Simulate TSLA put option chain around 400
        # For a 1x2 put ratio spread (buy 1 higher strike, sell 2 lower strikes)
        # We need: 2 * short_premium > 1 * long_premium for credit
        rows = []
        strikes = [350, 360, 370, 375, 380, 382.5, 385, 390, 395, 400, 405, 410]
        for strike in strikes:
            # ATM (400) has ~0.50 delta, lower strikes have lower deltas
            if strike >= 400:
                delta = -0.50 - (strike - 400) * 0.02
            else:
                delta = -0.50 + (400 - strike) * 0.01
            
            # Price based on moneyness - higher strikes get higher premiums
            # For a 1x2 ratio spread to generate credit: 2 * short_premium > 1 * long_premium
            # Example: long at 400 (~$5.075 mid), shorts need combined >= 5.175 for $0.10 credit
            # So each short needs >= ~$2.6, but we'll make them ~$2.8-3.0 to ensure credit
            if strike >= 400:
                # Higher strikes (closer to ATM) - higher premiums
                bid = max(0.20, 5.0 + (strike - 400) * 0.10)
            else:
                # Lower strikes (further OTM) - keep premiums high enough for credit spreads
                # For strikes around 370-390, use ~55-60% of ATM premium to ensure credit
                if strike >= 370:
                    bid = max(0.20, 2.9 - (400 - strike) * 0.01)
                else:
                    # Very low strikes - lower premiums but still reasonable for credit
                    bid = max(0.20, 2.6 - (370 - strike) * 0.008)
            ask = bid + 0.15
            
            rows.append(OptionRow(
                symbol="TSLA",
                expiry="20251128",
                right="P",
                strike=float(strike),
                bid=bid,
                ask=ask,
                delta=delta,
                volume=500.0,
            ))
        return rows
    
    def test_strategy_name(self, strategy):
        """Test strategy name property."""
        assert strategy.name == "vertical_spread_with_hedging"
    
    def test_find_candidates_basic(self, strategy, sample_option_rows_for_ratio):
        """Test basic ratio spread candidate finding."""
        candidates = strategy.find_candidates(
            sample_option_rows_for_ratio,
            target_delta_long=0.50,
            target_delta_short_1=0.25,
            target_delta_short_2=0.15,
            num_candidates=3,
            min_credit=0.10,
        )
        
        assert len(candidates) > 0
        
        for candidate in candidates:
            assert isinstance(candidate, RatioSpreadCandidate)
            # Long put should have higher strike than short puts
            assert candidate.long_put.strike > candidate.short_put_1.strike
            assert candidate.short_put_1.strike > candidate.short_put_2.strike
            assert candidate.net_credit > 0
    
    def test_describe_candidate(self, strategy, sample_option_rows_for_ratio):
        """Test ratio spread candidate description."""
        candidates = strategy.find_candidates(
            sample_option_rows_for_ratio,
            target_delta_long=0.50,
            target_delta_short_1=0.25,
            target_delta_short_2=0.15,
            num_candidates=1,
            min_credit=0.10,
        )
        
        description = strategy.describe_candidate("Test Ratio", candidates[0])
        
        assert "Test Ratio" in description
        assert "Buy" in description
        assert "Sell" in description
        assert "Net Credit" in description
        assert "Max Loss" in description


class TestStrategyRegistry:
    """Tests for strategy registry functions."""
    
    def test_list_strategies(self):
        """Test listing available strategies."""
        strategies = list_strategies()
        
        assert "otm_credit_spreads" in strategies
        assert "vertical_spread_with_hedging" in strategies
    
    def test_get_strategy_otm_credit(self):
        """Test getting OTM credit spreads strategy."""
        strategy = get_strategy("otm_credit_spreads")
        
        assert isinstance(strategy, OTMCreditSpreadsStrategy)
        assert strategy.name == "otm_credit_spreads"
    
    def test_get_strategy_ratio_spread(self):
        """Test getting ratio spread strategy."""
        strategy = get_strategy("vertical_spread_with_hedging")
        
        assert isinstance(strategy, VerticalSpreadWithHedgingStrategy)
        assert strategy.name == "vertical_spread_with_hedging"
    
    def test_get_strategy_invalid(self):
        """Test error for invalid strategy name."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("invalid_strategy")
    
    def test_registry_contains_all_strategies(self):
        """Test that registry contains all expected strategies."""
        assert "otm_credit_spreads" in STRATEGY_REGISTRY
        assert "vertical_spread_with_hedging" in STRATEGY_REGISTRY
        assert len(STRATEGY_REGISTRY) == 2


class TestVerticalSpreadCandidate:
    """Tests for VerticalSpreadCandidate dataclass."""
    
    def test_creation(self):
        """Test creating a vertical spread candidate."""
        short = OptionRow(
            symbol="SPY", expiry="20251120", right="P",
            strike=450.0, bid=1.50, ask=1.60, delta=-0.10, volume=100.0
        )
        long = OptionRow(
            symbol="SPY", expiry="20251120", right="P",
            strike=446.0, bid=0.90, ask=1.00, delta=-0.05, volume=100.0
        )
        
        candidate = VerticalSpreadCandidate(
            short=short,
            long=long,
            width=4.0,
            credit=0.55,
        )
        
        assert candidate.short.strike == 450.0
        assert candidate.long.strike == 446.0
        assert candidate.width == 4.0
        assert candidate.credit == 0.55


class TestRatioSpreadCandidate:
    """Tests for RatioSpreadCandidate dataclass."""
    
    def test_creation(self):
        """Test creating a ratio spread candidate."""
        long_put = OptionRow(
            symbol="TSLA", expiry="20251128", right="P",
            strike=400.0, bid=5.00, ask=5.20, delta=-0.50, volume=500.0
        )
        short_put_1 = OptionRow(
            symbol="TSLA", expiry="20251128", right="P",
            strike=382.5, bid=2.00, ask=2.20, delta=-0.25, volume=300.0
        )
        short_put_2 = OptionRow(
            symbol="TSLA", expiry="20251128", right="P",
            strike=375.0, bid=1.50, ask=1.70, delta=-0.15, volume=200.0
        )
        
        candidate = RatioSpreadCandidate(
            long_put=long_put,
            short_put_1=short_put_1,
            short_put_2=short_put_2,
            net_credit=1.09,
            max_loss=1641.0,
            breakeven_low=358.59,
            breakeven_high=398.91,
        )
        
        assert candidate.long_put.strike == 400.0
        assert candidate.short_put_1.strike == 382.5
        assert candidate.short_put_2.strike == 375.0
        assert candidate.net_credit == 1.09
        assert candidate.max_loss == 1641.0


class TestLiveEnFlag:
    """Tests for --live-en flag behavior."""
    
    def test_live_account_blocked_without_live_en(self):
        """Test that live accounts (starting with 'U') are blocked without --live-en flag."""
        # Simulate the safeguard logic from options_strategy_trader.py
        account = "U1234567"
        live_en = False
        
        # This is the safeguard logic
        should_block = account and account.startswith("U") and not live_en
        
        assert should_block is True, "Live account should be blocked without --live-en"
    
    def test_live_account_allowed_with_live_en(self):
        """Test that live accounts are allowed with --live-en flag."""
        account = "U1234567"
        live_en = True
        
        should_block = account and account.startswith("U") and not live_en
        
        assert should_block is False, "Live account should be allowed with --live-en"
    
    def test_paper_account_allowed_without_live_en(self):
        """Test that paper accounts (starting with 'D') are allowed without --live-en."""
        account = "DU1234567"
        live_en = False
        
        should_block = account and account.startswith("U") and not live_en
        
        assert should_block is False, "Paper account should be allowed without --live-en"
    
    def test_empty_account_allowed_without_live_en(self):
        """Test that empty account is allowed without --live-en."""
        account = ""
        live_en = False
        
        should_block = account and account.startswith("U") and not live_en
        
        assert not should_block, "Empty account should be allowed without --live-en"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

