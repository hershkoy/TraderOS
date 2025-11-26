#!/usr/bin/env python3
"""
option_strategies.py

Strategy classes for selecting options and building spread candidates.
Each strategy class handles the logic for finding and constructing specific option strategies.
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class OptionRow:
    """Represents a single option from the chain."""
    symbol: str
    expiry: str
    right: str
    strike: float
    bid: float
    ask: float
    delta: Optional[float]
    volume: Optional[float]

    @property
    def mid(self) -> Optional[float]:
        if math.isnan(self.bid) or math.isnan(self.ask):
            return None
        if self.bid == "" or self.ask == "":
            return None
        try:
            return (float(self.bid) + float(self.ask)) / 2.0
        except Exception:
            return None


@dataclass
class SpreadCandidate:
    """Base class for spread candidates."""
    short: OptionRow
    long: OptionRow
    width: float
    credit: float  # estimated mid credit (or debit if negative)


@dataclass
class VerticalSpreadCandidate(SpreadCandidate):
    """A simple vertical spread (2 legs)."""
    pass


@dataclass
class RatioSpreadCandidate:
    """
    A ratio spread with hedging (3 legs).
    Example: Buy 1 ATM put, Sell 2 OTM puts (put ratio spread / backspread).
    Based on the image: Buy 400P, Sell 382.5P, Sell 375P
    """
    long_put: OptionRow  # The bought put (higher strike, e.g., 400P)
    short_put_1: OptionRow  # First sold put (e.g., 382.5P)
    short_put_2: OptionRow  # Second sold put (e.g., 375P)
    net_credit: float  # Net credit received
    max_loss: float  # Maximum loss (occurs between short strikes)
    breakeven_low: float  # Lower breakeven
    breakeven_high: float  # Upper breakeven


class OptionStrategy(ABC):
    """Abstract base class for option strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name identifier."""
        pass
    
    @abstractmethod
    def find_candidates(
        self,
        rows: List[OptionRow],
        **kwargs
    ) -> List:
        """Find spread candidates based on strategy criteria."""
        pass
    
    @abstractmethod
    def describe_candidate(self, label: str, candidate) -> str:
        """Generate human-readable description of a candidate."""
        pass


class OTMCreditSpreadsStrategy(OptionStrategy):
    """
    OTM Credit Spreads Strategy.
    
    Sells an OTM put and buys a further OTM put for protection.
    Target: ~10 delta short leg, collect premium.
    """
    
    @property
    def name(self) -> str:
        return "otm_credit_spreads"
    
    def find_candidates(
        self,
        rows: List[OptionRow],
        width: float = 4.0,
        target_delta: float = 0.10,
        num_candidates: int = 3,
        min_credit: float = 0.15,
        **kwargs
    ) -> List[VerticalSpreadCandidate]:
        """
        Find num_candidates vertical put spreads with short leg around target_delta.
        Short put delta is negative -> we use abs(delta).
        Long put is width points lower in strike.
        """
        if target_delta is None:
            target_delta = 0.10
        
        strike_map = {r.strike: r for r in rows}
        
        # Filter rows with usable greeks + quotes
        usable = []
        for r in rows:
            if r.delta is None or math.isnan(r.delta):
                continue
            if r.mid is None:
                continue
            d_abs = abs(r.delta)
            if d_abs < 0.02 or d_abs > 0.35:
                continue
            usable.append(r)
        
        if not usable:
            total_with_delta = sum(1 for r in rows if r.delta is not None and not math.isnan(r.delta))
            total_with_mid = sum(1 for r in rows if r.mid is not None)
            raise ValueError(
                f"No usable options with delta and quotes. "
                f"Found {total_with_delta} options with delta, {total_with_mid} with mid prices, "
                f"but none in delta range [0.02, 0.35]"
            )
        
        logger.info(f"Found {len(usable)} usable options with delta and quotes")
        
        usable = [r for r in usable if r.delta is not None and not math.isnan(r.delta)]
        usable.sort(key=lambda r: abs(abs(r.delta) - target_delta))
        
        candidates: List[VerticalSpreadCandidate] = []
        skipped_no_long = 0
        skipped_low_credit = 0
        
        for r in usable:
            long_strike = r.strike - width
            long_row = strike_map.get(long_strike)
            actual_width = width
            
            if not long_row or long_row.mid is None:
                available_strikes = [s for s in strike_map.keys() if s < r.strike]
                if available_strikes:
                    available_strikes.sort(key=lambda s: abs((r.strike - s) - width))
                    for candidate_long_strike in available_strikes:
                        actual_width_candidate = r.strike - candidate_long_strike
                        if actual_width_candidate >= width * 0.5:
                            long_row = strike_map.get(candidate_long_strike)
                            if long_row and long_row.mid is not None:
                                long_strike = candidate_long_strike
                                actual_width = actual_width_candidate
                                if abs(actual_width - width) > 0.5:
                                    logger.debug(f"Using adjusted width: {r.strike}/{long_strike} (width={actual_width:.1f} instead of {width:.1f})")
                                break
            
            if not long_row or long_row.mid is None:
                skipped_no_long += 1
                continue
            
            credit = r.mid - long_row.mid
            if credit < min_credit:
                skipped_low_credit += 1
                logger.debug(f"Skipping {r.strike}/{long_strike} spread: credit ${credit:.2f} < min ${min_credit:.2f}")
                continue
            
            candidates.append(VerticalSpreadCandidate(short=r, long=long_row, width=actual_width, credit=credit))
            if len(candidates) >= num_candidates:
                break
        
        if not candidates:
            logger.warning(f"Skipped {skipped_no_long} candidates: long strike not found")
            logger.warning(f"Skipped {skipped_low_credit} candidates: credit too low (min: ${min_credit:.2f})")
            
            sample_credits = []
            for r in usable[:5]:
                long_strike = r.strike - width
                long_row = strike_map.get(long_strike)
                if long_row and long_row.mid is not None:
                    credit = r.mid - long_row.mid
                    sample_credits.append(credit)
            
            if sample_credits:
                max_credit = max(sample_credits)
                min_credit_found = min(sample_credits)
                raise ValueError(
                    f"No spread candidates met the filters. "
                    f"Found {len(usable)} usable short legs, but:\n"
                    f"  - {skipped_no_long} skipped: long strike not found (width={width})\n"
                    f"  - {skipped_low_credit} skipped: credit < ${min_credit:.2f}\n"
                    f"  - Sample credit range: ${min_credit_found:.2f} - ${max_credit:.2f}\n"
                    f"  - Try: --min-credit {max(0.05, min_credit_found - 0.05):.2f} or --spread-width {width + 1}"
                )
            else:
                raise ValueError(
                    f"No spread candidates met the filters. "
                    f"Found {len(usable)} usable short legs, but none had matching long strikes. "
                    f"Try adjusting --spread-width (current: {width})"
                )
        
        return candidates
    
    def describe_candidate(self, label: str, candidate: VerticalSpreadCandidate) -> str:
        d_abs = abs(candidate.short.delta) if candidate.short.delta is not None and not math.isnan(candidate.short.delta) else None
        lines = [
            f"{label}",
            "",
            f"Sell {candidate.short.strike:.0f} {candidate.short.right} / Buy {candidate.long.strike:.0f} {candidate.long.right} "
            f"({int(candidate.width)}-wide)",
        ]
        if d_abs is not None:
            lines.append(f"Delta ~ {candidate.short.delta:.3f} (|delta| ~ {d_abs:.3f})")
        lines.append(f"Credit ~ ${candidate.credit:.2f}")
        if candidate.short.volume and not math.isnan(candidate.short.volume):
            lines.append(f"Short leg volume: {int(candidate.short.volume)}")
        return "\n".join(lines)


class VerticalSpreadWithHedgingStrategy(OptionStrategy):
    """
    Vertical Spread with Hedging (Put Ratio Spread).
    
    Based on the provided image:
    - Buy 1 higher strike put (e.g., 400P ATM)
    - Sell 2 lower strike puts (e.g., 382.5P and 375P)
    
    This creates a position that:
    - Profits if stock stays above upper breakeven
    - Has limited loss between short strikes
    - Has unlimited risk below lower breakeven (unless hedged)
    
    The image shows: Net Credit $109, Max Loss $1,641, Breakevens $358.59 - $398.91
    """
    
    @property
    def name(self) -> str:
        return "vertical_spread_with_hedging"
    
    def find_candidates(
        self,
        rows: List[OptionRow],
        target_delta_long: float = 0.50,  # ATM put
        target_delta_short_1: float = 0.25,  # First short put
        target_delta_short_2: float = 0.15,  # Second short put (further OTM)
        num_candidates: int = 3,
        min_credit: float = 0.50,
        **kwargs
    ) -> List[RatioSpreadCandidate]:
        """
        Find ratio spread candidates (1x2 put ratio spread).
        Buy 1 higher strike put, sell 2 lower strike puts.
        """
        strike_map = {r.strike: r for r in rows}
        
        # Filter rows with usable greeks + quotes
        usable = []
        for r in rows:
            if r.delta is None or math.isnan(r.delta):
                continue
            if r.mid is None:
                continue
            usable.append(r)
        
        if not usable:
            raise ValueError("No usable options with delta and quotes found")
        
        logger.info(f"Found {len(usable)} usable options for ratio spread")
        
        # Debug: show delta distribution
        deltas = [abs(r.delta) for r in usable]
        logger.debug(f"Delta range: {min(deltas):.4f} to {max(deltas):.4f}")
        logger.debug(f"Sample strikes/deltas: {[(r.strike, abs(r.delta)) for r in usable[:10]]}")
        
        # Structure: Sell 1 OTM put, Buy 2 further OTM puts (1x2 put ratio backspread for credit)
        # This creates a credit when sold put premium > 2x bought put premiums
        
        # Find short put candidates (OTM, delta ~0.15-0.25)
        short_candidates = sorted(
            [r for r in usable if abs(r.delta) >= 0.10 and abs(r.delta) <= 0.30],
            key=lambda r: abs(abs(r.delta) - target_delta_short_1)
        )
        
        logger.debug(f"Short candidates (delta 0.10-0.30): {len(short_candidates)}")
        
        if not short_candidates:
            raise ValueError(f"No suitable short put candidates found (target delta: {target_delta_short_1})")
        
        candidates: List[RatioSpreadCandidate] = []
        
        for short_put in short_candidates[:10]:  # Check top 10 short candidates
            # Find long put 1 (further OTM than short_put, delta ~0.05-0.10)
            long_1_candidates = sorted(
                [r for r in usable if r.strike < short_put.strike and abs(r.delta) >= 0.03 and abs(r.delta) <= 0.15],
                key=lambda r: abs(abs(r.delta) - target_delta_short_2)
            )
            
            if len(long_1_candidates) < 2:
                continue
            
            for i, long_put_1 in enumerate(long_1_candidates[:5]):
                # Find long put 2 (even further OTM or same strike as long_put_1)
                long_2_candidates = [r for r in long_1_candidates if r.strike <= long_put_1.strike and r != long_put_1]
                
                if not long_2_candidates:
                    # Use same strike for both longs (buy 2 at same strike)
                    long_put_2 = long_put_1
                else:
                    long_put_2 = long_2_candidates[0]
                
                # Calculate net credit: short premium - (long_1 + long_2 premiums)
                net_credit = short_put.mid - long_put_1.mid - long_put_2.mid
                
                if net_credit < min_credit:
                    logger.debug(f"Skipping: short {short_put.strike} - long {long_put_1.strike}/{long_put_2.strike} = ${net_credit:.2f} < min ${min_credit:.2f}")
                    continue
                
                # Calculate max loss (occurs between short and long strikes)
                # Max loss = (short_strike - long_1_strike) * 100 - net_credit * 100
                max_loss = (short_put.strike - long_put_1.strike) * 100 - net_credit * 100
                
                # Calculate breakevens
                # Upper breakeven: short_strike - net_credit
                breakeven_high = short_put.strike - net_credit
                # Lower breakeven: long strikes provide protection below
                breakeven_low = long_put_2.strike - (max_loss / 100)
                
                candidate = RatioSpreadCandidate(
                    long_put=long_put_1,  # Primary long
                    short_put_1=short_put,  # The sold put
                    short_put_2=long_put_2,  # Secondary long (reusing field)
                    net_credit=net_credit,
                    max_loss=max_loss,
                    breakeven_low=breakeven_low,
                    breakeven_high=breakeven_high
                )
                candidates.append(candidate)
                logger.debug(f"Found candidate: sell {short_put.strike}, buy {long_put_1.strike}/{long_put_2.strike}, credit=${net_credit:.2f}")
                
                if len(candidates) >= num_candidates:
                    break
            
            if len(candidates) >= num_candidates:
                break
        
        if not candidates:
            raise ValueError(
                f"No ratio spread candidates found meeting criteria. "
                f"Try adjusting delta targets or min_credit (current: ${min_credit:.2f})"
            )
        
        return candidates
    
    def describe_candidate(self, label: str, candidate: RatioSpreadCandidate) -> str:
        lines = [
            f"{label}",
            "",
            f"Sell 1x {candidate.short_put_1.strike:.1f} {candidate.short_put_1.right}",
            f"Buy 1x {candidate.long_put.strike:.1f} {candidate.long_put.right}",
            f"Buy 1x {candidate.short_put_2.strike:.1f} {candidate.short_put_2.right}",
            "",
            f"Net Credit ~ ${candidate.net_credit:.2f}",
            f"Max Loss ~ ${candidate.max_loss:.2f}",
            f"Breakevens: ${candidate.breakeven_low:.2f} - ${candidate.breakeven_high:.2f}",
        ]
        
        if candidate.long_put.delta:
            lines.append(f"Long delta: {candidate.long_put.delta:.3f}")
        if candidate.short_put_1.delta:
            lines.append(f"Short 1 delta: {candidate.short_put_1.delta:.3f}")
        if candidate.short_put_2.delta:
            lines.append(f"Short 2 delta: {candidate.short_put_2.delta:.3f}")
        
        return "\n".join(lines)


# Strategy registry
STRATEGY_REGISTRY: Dict[str, type] = {
    "otm_credit_spreads": OTMCreditSpreadsStrategy,
    "vertical_spread_with_hedging": VerticalSpreadWithHedgingStrategy,
}


def get_strategy(strategy_name: str) -> OptionStrategy:
    """Get strategy instance by name."""
    if strategy_name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {available}")
    return STRATEGY_REGISTRY[strategy_name]()


def list_strategies() -> List[str]:
    """List available strategy names."""
    return list(STRATEGY_REGISTRY.keys())

