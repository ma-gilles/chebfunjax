"""Port of MATLAB Chebfun tests/chebtech2/test_constructor.m (Opus 4.8).

MATLAB's ``test_constructor`` exercises the non-user-facing ``populate()``
with ``pref.extrapolate`` (0/1), ``pref.refinementFunction``
('nested' / 'resampling'), NaN/Inf error handling, an extrapolation
endpoint-avoidance test, ``minSamples``/``maxLength`` prefs and
logical-valued construction.  chebfunjax has none of that machinery:
``from_function`` is a single adaptive constructor (extrapolate OFF, one
refinement path) with no prefs.

FIXED (Fable 5): ``from_function`` now accepts ``extrapolate=`` (MATLAB
``pref.extrapolate``: evaluate interior points only, extrapolate the endpoints)
and constructs array-valued techs, and its adaptive path resamples the whole
grid each iteration (MATLAB ``refinementFunction='resampling'``), so passes
3-15 hold.  ``pref.minSamples``/``pref.maxLength`` are still unsupported, so
pass(16) stays skipped.

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
        # MATLAB pass(3): extrapolate=1, scalar sin.  FIXED (Fable 5).
        g = Chebtech2.from_function(jnp.sin, extrapolate=True)
        x = chebpts(len(g.coeffs), kind=2)
        values = Chebtech2.coeffs2vals(g.coeffs)
        assert _ninf(jnp.sin(x) - values) < TOL

    def test_scalar_sin_nested_extrap1_vscale(self):
        # MATLAB pass(4): extrapolate=1 -> vscale within tol of sin(1)
        # (100*eps, not eps: the endpoint is extrapolated).  FIXED (Fable 5).
        g = Chebtech2.from_function(jnp.sin, extrapolate=True)
        assert abs(g.vscale - float(np.sin(1.0))) < TOL

    def test_scalar_sin_resampling_extrap0_accuracy(self):
        # MATLAB pass(5): the chebfunjax adaptive path resamples the whole grid
        # each iteration (== refinementFunction='resampling').  FIXED (Fable 5).
        g = Chebtech2.from_function(jnp.sin)
        x = chebpts(len(g.coeffs), kind=2)
        values = Chebtech2.coeffs2vals(g.coeffs)
        assert _ninf(jnp.sin(x) - values) < TOL

    def test_scalar_sin_resampling_extrap0_vscale(self):
        # MATLAB pass(6): endpoints sampled -> vscale == sin(1) to eps.
        g = Chebtech2.from_function(jnp.sin)
        assert abs(g.vscale - float(np.sin(1.0))) < EPS

    def test_scalar_sin_resampling_extrap1_accuracy(self):
        # MATLAB pass(7): resampling + extrapolate=1.  FIXED (Fable 5).
        g = Chebtech2.from_function(jnp.sin, extrapolate=True)
        x = chebpts(len(g.coeffs), kind=2)
        values = Chebtech2.coeffs2vals(g.coeffs)
        assert _ninf(jnp.sin(x) - values) < TOL

    def test_scalar_sin_resampling_extrap1_vscale(self):
        # MATLAB pass(8): resampling + extrapolate=1 -> vscale within tol.
        g = Chebtech2.from_function(jnp.sin, extrapolate=True)
        assert abs(g.vscale - float(np.sin(1.0))) < TOL

    @staticmethod
    def _array_op(x):
        return jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)

    def test_array_nested_extrap0(self):
        # MATLAB pass(9): array-valued [sin cos exp], extrapolate=0.
        # FIXED (Fable 5, Big-Three array-valued epic).
        g = Chebtech2.from_function(self._array_op)
        x = chebpts(g.coeffs.shape[0], kind=2)
        values = Chebtech2.coeffs2vals(g.coeffs)
        assert _ninf(self._array_op(x) - values) < TOL

    def test_array_nested_extrap1(self):
        # MATLAB pass(10): array-valued, extrapolate=1.  FIXED (Fable 5).
        g = Chebtech2.from_function(self._array_op, extrapolate=True)
        x = chebpts(g.coeffs.shape[0], kind=2)
        values = Chebtech2.coeffs2vals(g.coeffs)
        assert _ninf(self._array_op(x) - values) < TOL

    def test_array_resampling_extrap0(self):
        # MATLAB pass(11): array-valued, resampling, extrapolate=0.
        g = Chebtech2.from_function(self._array_op)
        x = chebpts(g.coeffs.shape[0], kind=2)
        values = Chebtech2.coeffs2vals(g.coeffs)
        assert _ninf(self._array_op(x) - values) < TOL

    def test_array_resampling_extrap1(self):
        # MATLAB pass(12): array-valued, resampling, extrapolate=1.
        g = Chebtech2.from_function(self._array_op, extrapolate=True)
        x = chebpts(g.coeffs.shape[0], kind=2)
        values = Chebtech2.coeffs2vals(g.coeffs)
        assert _ninf(self._array_op(x) - values) < TOL

    def test_nan_raises(self):
        # MATLAB pass(13): x + NaN -> 'Too many NaNs/Infs to handle.'
        # FIXED (Fable 5): constructor extrapolation raises on all-NaN samples.
        with pytest.raises(Exception):
            Chebtech2.from_function(lambda x: x + jnp.nan, n=17)

    def test_inf_raises(self):
        # MATLAB pass(14): x + Inf -> 'Too many NaNs/Infs to handle.'
        with pytest.raises(Exception):
            Chebtech2.from_function(lambda x: x + jnp.inf, n=17)

    def test_extrapolate_avoids_endpoints(self):
        # MATLAB pass(15): extrapolate=1 must not evaluate f at |x| == 1.
        # FIXED (Fable 5): interior-only sampling + endpoint extrapolation.
        def F(x):
            if bool(np.any(np.abs(np.asarray(x)) == 1.0)):
                raise RuntimeError("Extrapolate should prevent endpoint evaluation.")
            return jnp.sin(x)

        # Must not raise:
        Chebtech2.from_function(
            lambda x: jnp.stack([F(x), F(x)], axis=-1), extrapolate=True
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
