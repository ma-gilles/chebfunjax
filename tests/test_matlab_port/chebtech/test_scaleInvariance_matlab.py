"""Port of MATLAB Chebfun tests/chebtech/test_scaleInvariance.m (Opus 4.8).

Vertical-scale invariance of chebtech construction: constructing ``scale*F``
must give exactly ``scale`` times the coefficients of constructing ``F`` (and
similarly for ``F/scale``).  MATLAB checks ``~any(f.coeffs - f1.coeffs/scale)``
i.e. the coefficient vectors are bit-identical.  We reproduce that exactly
(after prolonging to a common length; adaptive construction picks the same
length here, so prolong is a no-op).

The MATLAB test runs the same two checks twice, once with
``pref.happinessCheck = 'classic'`` and once with ``'strict'``.  chebfunjax has
no such pref, and the math is identical, so we port ONE faithful version
(pass 1 & 2) and skip the strict-pref duplicates (pass 3 & 4).

Provenance
----------
MATLAB source : tests/chebtech/test_scaleInvariance.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

CASES = [(Chebtech1, 1), (Chebtech2, 2)]

SCALE = 2.0 ** 300


def _F(x):
    return jnp.sin(10000 * x)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _aligned_diff(a, b):
    # Prolong both coefficient vectors to a common length before comparing.
    n = max(a.shape[0], b.shape[0])
    fa = Chebtech2.from_coeffs(a).prolong(n).coeffs
    fb = Chebtech2.from_coeffs(b).prolong(n).coeffs
    return _ninf(fa - fb)


class TestChebtechScaleInvariance:
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_multiply_by_scale(self, Tech, kind):
        # pass(n, 1): scale*chebtech(F) == chebtech(scale*F) in coeffs (exact).
        f = Tech.from_function(_F)
        f1 = Tech.from_function(lambda x: _F(x) * SCALE)
        assert _aligned_diff(f.coeffs, f1.coeffs / SCALE) == 0.0

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_divide_by_scale(self, Tech, kind):
        # pass(n, 2): chebtech(F)/scale == chebtech(F/scale) in coeffs (exact).
        f = Tech.from_function(_F)
        f2 = Tech.from_function(lambda x: _F(x) / SCALE)
        assert _aligned_diff(f.coeffs, f2.coeffs * SCALE) == 0.0

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_multiply_by_scale_strict_pref(self, Tech, kind):
        # pass(n, 3): identical to pass(n, 1) but with happinessCheck='strict'.
        pytest.skip(
            "chebfunjax has no happinessCheck='strict' pref; identical math to "
            "test_multiply_by_scale already ported"
        )

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_divide_by_scale_strict_pref(self, Tech, kind):
        # pass(n, 4): identical to pass(n, 2) but with happinessCheck='strict'.
        pytest.skip(
            "chebfunjax has no happinessCheck='strict' pref; identical math to "
            "test_divide_by_scale already ported"
        )
