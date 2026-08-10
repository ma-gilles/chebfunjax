"""Port of MATLAB Chebfun tests/chebtech/test_scaleInvariance.m (Opus 4.8).

Vertical-scale invariance of chebtech construction: constructing ``scale*F``
must give exactly ``scale`` times the coefficients of constructing ``F`` (and
similarly for ``F/scale``).  MATLAB checks ``~any(f.coeffs - f1.coeffs/scale)``
i.e. the coefficient vectors are bit-identical.  We reproduce that exactly
(after prolonging to a common length; adaptive construction picks the same
length here, so prolong is a no-op).

The MATLAB test runs the same two checks twice, once with
``pref.happinessCheck = 'classic'`` and once with ``'strict'``.  chebfunjax
exposes that preference as ``from_function(..., check=...)``, so all four
passes are ported with the same prefs MATLAB sets.

Note: ``strictCheck`` requires every tail coefficient to fall below
``eps*vscale`` with no length relaxation, which ``sin(10000*x)`` does not
achieve at any grid size up to the maximum; construction therefore returns an
unhappy maximum-length representation (as MATLAB's does).  Scale invariance
still holds bit-exactly there, which is all passes 3-4 assert.

No gaps: all four MATLAB passes are exercised.

Provenance
----------
MATLAB source : tests/chebtech/test_scaleInvariance.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

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
        # MATLAB sets pref.happinessCheck = 'classic' for passes 1-2.
        f = Tech.from_function(_F, check="classic")
        f1 = Tech.from_function(lambda x: _F(x) * SCALE, check="classic")
        assert _aligned_diff(f.coeffs, f1.coeffs / SCALE) == 0.0

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_divide_by_scale(self, Tech, kind):
        # pass(n, 2): chebtech(F)/scale == chebtech(F/scale) in coeffs (exact).
        f = Tech.from_function(_F, check="classic")
        f2 = Tech.from_function(lambda x: _F(x) / SCALE, check="classic")
        assert _aligned_diff(f.coeffs, f2.coeffs * SCALE) == 0.0

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_multiply_by_scale_strict_pref(self, Tech, kind):
        # pass(n, 3): as pass(n, 1) but with happinessCheck = 'strict'.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # strictCheck never converges here
            f = Tech.from_function(_F, check="strict")
            f1 = Tech.from_function(lambda x: SCALE * _F(x), check="strict")
        assert _aligned_diff(f.coeffs, f1.coeffs / SCALE) == 0.0

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_divide_by_scale_strict_pref(self, Tech, kind):
        # pass(n, 4): as pass(n, 2) but with happinessCheck = 'strict'.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # strictCheck never converges here
            f = Tech.from_function(_F, check="strict")
            f2 = Tech.from_function(lambda x: _F(x) / SCALE, check="strict")
        assert _aligned_diff(f.coeffs, f2.coeffs * SCALE) == 0.0
