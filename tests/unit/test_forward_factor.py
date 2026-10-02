"""Unit tests for Forward Factor math helpers."""
from __future__ import annotations

import math

from utils.research.forward_factor import forward_factor, forward_vol


def test_forward_vol_video_example():
    # Video / blog: 30d IV 45%, 60d IV 35% -> fwd ~20.62%, FF ~1.18
    iv1, iv2 = 0.45, 0.35
    t1, t2 = 30 / 365.0, 60 / 365.0
    fv = forward_vol(iv1, iv2, t1, t2)
    assert fv is not None
    assert abs(fv - 0.2062) < 0.002
    ff = forward_factor(iv1, fv)
    assert ff is not None
    assert abs(ff - 1.18) < 0.03


def test_forward_vol_rejects_crossed_variance():
    # Back IV too low vs front can make forward variance negative
    assert forward_vol(0.50, 0.10, 30 / 365.0, 60 / 365.0) is None


def test_forward_factor_zero_when_flat():
    iv = 0.20
    t1, t2 = 60 / 365.0, 90 / 365.0
    fv = forward_vol(iv, iv, t1, t2)
    assert fv is not None
    assert abs(fv - iv) < 1e-9
    assert abs(forward_factor(iv, fv)) < 1e-9
