"""CLI suite tags for the bull put claim-check."""
from __future__ import annotations

from scripts.research.claimcheck_spx_bull_put import VARIANT_SUITES, _tag


def test_baseline_suite_covers_slip_and_stack():
    tags = [_tag(cfg) for cfg in VARIANT_SUITES["baseline"]]
    assert len(tags) == len(set(tags))
    assert "mid_slip0.05_sl2x" in tags
    assert "mid_slip0.15_sl2x" in tags
    assert "ask_bid_sl2x" in tags
    assert "mid_slip0.05_nostop" in tags
    assert "mid_slip0.05_sl2x_c4" in tags
