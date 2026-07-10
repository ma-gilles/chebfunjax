"""Port of MATLAB Chebfun tests/chebtech2/test_constructor.m (Opus 4.8).

MATLAB's ``test_constructor`` exercises the non-user-facing ``populate()``
with ``pref.extrapolate`` (0/1), ``pref.refinementFunction``
('nested' / 'resampling'), NaN/Inf error handling, an extrapolation
endpoint-avoidance test, ``minSamples``/``maxLength`` prefs and
logical-valued construction.  chebfunjax has none of that machinery:
``from_function`` is a single adaptive constructor (extrapolate OFF, one
refinement path) with no prefs.

The faithfully portable assertions are the two scalar ``sin`` checks for
the default case (``extrapolate=0``, nested refinement):
  * pass(1): construction accuracy at ``tol = 100*eps``
  * pass(2): ``vscale(g) == sin(1)`` (to eps) and ``g.ishappy``
Everything that turns on a pref chebfunjax lacks is skipped precisely.

Provenance
----------
MATLAB source : tests/chebtech2/test_constructor.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.utils.quadrature import chebpts

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestChebtech2Constructor:
    def test_scalar_sin_nested_extrap0_accuracy(self):
        # MATLAB pass(1): extrapolate=0, refinementFunction='nested', scalar sin.
        g = Chebtech2.from_function(jnp.sin)
        x = chebpts(len(g.coeffs), kind=2)
        values = Chebtech2.coeffs2vals(g.coeffs)
        assert _ninf(jnp.sin(x) - values) < TOL

    def test_scalar_sin_nested_extrap0_vscale(self):
        # MATLAB pass(2): abs(vscale(g) - sin(1)) < eps && g.ishappy && eps < tol.
        g = Chebtech2.from_function(jnp.sin)
        assert abs(g.vscale - float(np.sin(1.0))) < EPS
        assert g.ishappy
        assert EPS < TOL

    def test_scalar_sin_nested_extrap1_accuracy(self):
        pytest.skip("chebfunjax has no pref.extrapolate option")

    def test_scalar_sin_nested_extrap1_vscale(self):
        pytest.skip("chebfunjax has no pref.extrapolate option")

    def test_scalar_sin_resampling_extrap0_accuracy(self):
        pytest.skip(
            "chebfunjax has no pref.refinementFunction='resampling' "
            "(single adaptive construction path)"
        )

    def test_scalar_sin_resampling_extrap0_vscale(self):
        pytest.skip(
            "chebfunjax has no pref.refinementFunction='resampling' "
            "(single adaptive construction path)"
        )

    def test_scalar_sin_resampling_extrap1_accuracy(self):
        pytest.skip("chebfunjax has no pref.extrapolate/refinementFunction options")

    def test_scalar_sin_resampling_extrap1_vscale(self):
        pytest.skip("chebfunjax has no pref.extrapolate/refinementFunction options")

    def test_array_nested_extrap0(self):
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix techs"
        )

    def test_array_nested_extrap1(self):
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix techs"
        )

    def test_array_resampling_extrap0(self):
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix techs"
        )

    def test_array_resampling_extrap1(self):
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix techs"
        )

    def test_nan_raises(self):
        pytest.skip(
            "chebfunjax has no populate() NaN/Inf handling "
            "('Too many NaNs/Infs to handle.' error)"
        )

    def test_inf_raises(self):
        pytest.skip(
            "chebfunjax has no populate() NaN/Inf handling "
            "('Too many NaNs/Infs to handle.' error)"
        )

    def test_extrapolate_avoids_endpoints(self):
        pytest.skip(
            "chebfunjax has no pref.extrapolate; cannot avoid endpoint evaluation"
        )

    def test_minsamples_equals_maxlength(self):
        pytest.skip(
            "chebfunjax has no pref.minSamples/pref.maxLength construction options"
        )

    def test_logical_true(self):
        pytest.skip(
            "chebfunjax has no logical-valued construction / normest(); "
            "cannot port chebtech2(@(x) x > -2)"
        )

    def test_logical_false(self):
        pytest.skip(
            "chebfunjax has no logical-valued construction / normest(); "
            "cannot port chebtech2(@(x) x < -2)"
        )
