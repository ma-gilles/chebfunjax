"""Port of MATLAB Chebfun tests/chebtech1/test_constructor.m (Fable 5).

MATLAB's ``test_constructor`` exercises ``populate()``. The equivalent
adaptive entry point is ``Chebtech1.from_function``, including nested and
resampling strategies and minSamples/maxLength. ``normest`` is represented
by max(vscale), as in @chebtech/normest.m. Raw populate sample outputs for
passes3/4 are observed through the vectorized callback; the constructor's
sample probes are excluded from that observation.

Provenance
----------
MATLAB source : tests/chebtech1/test_constructor.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1
from chebfunjax.utils.quadrature import chebpts

EPS = float(np.finfo(np.float64).eps)


def _ninf(a):
    a = jnp.asarray(a)
    return float(jnp.max(jnp.sum(jnp.abs(a), axis=1) if a.ndim == 2 else jnp.abs(a)))


def _normest(g):
    """Equivalent of @chebtech/normest.m: ``out = max(vscale(f))``."""
    return float(np.max(np.asarray(g.vscale)))


class TestChebtech1Constructor:
    def test_scalar_sin_nested(self):
        # pass(1): populate with refinementFunction='nested', scalar sin.
        g = Chebtech1.from_function(jnp.sin)
        x = chebpts(len(g.coeffs), kind=1)
        values = Chebtech1.coeffs2vals(g.coeffs)
        assert _ninf(jnp.sin(x) - values) < 10 * g.vscale * EPS

    def test_array_sin_cos_exp_nested(self):
        # pass(2): array-valued [sin cos exp], tol 10*max(vscale*eps).
        fop = lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)
        g = Chebtech1.from_function(fop)
        assert g.coeffs.ndim == 2 and g.coeffs.shape[1] == 3
        x = chebpts(g.coeffs.shape[0], kind=1)
        values = Chebtech1.coeffs2vals(g.coeffs)
        assert _ninf(fop(x) - values) < 10 * _normest(g) * EPS

    def test_scalar_sin_resampling(self):
        # pass(3): MATLAB compares the unchopped VALUES returned by populate.
        sampled = []

        def op(x):
            values = jnp.sin(x)
            if x.size > 2:
                sampled.append(values)
            return values

        g = Chebtech1.from_function(op, refinement_function="resampling")
        values = sampled[-1]
        x = chebpts(values.shape[0], kind=1)
        assert _ninf(jnp.sin(x) - values) < 10 * g.vscale * EPS

    def test_array_sin_cos_exp_resampling(self):
        # pass(4): same populate-output assertion for three columns.
        sampled = []

        def op(x):
            values = jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)
            if x.size > 2:
                sampled.append(values)
            return values

        g = Chebtech1.from_function(op, refinement_function="resampling")
        values = sampled[-1]
        x = chebpts(values.shape[0], kind=1)
        exact = jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)
        assert _ninf(exact - values) < 10 * _normest(g) * EPS

    def test_nan_raises(self):
        # pass(5): @(x) x + NaN must error with 'Too many NaNs/Infs to
        # handle.'.  chebfunjax raises ValueError carrying that same message
        # (prefixed with the MATLAB-style identifier).
        with pytest.raises(ValueError, match="Too many NaNs/Infs to handle."):
            Chebtech1.from_function(lambda x: x + jnp.nan)

    def test_inf_raises(self):
        # pass(6): @(x) x + Inf must error with 'Too many NaNs/Infs to
        # handle.'.
        with pytest.raises(ValueError, match="Too many NaNs/Infs to handle."):
            Chebtech1.from_function(lambda x: x + jnp.inf)

    def test_minsamples_equals_maxlength(self):
        # pass(7): the real adaptive minSamples=maxLength=8 source options.
        # standardChop cannot resolve fewer than17 coefficients, so source
        # populate gives up; the original assertion only checks no crash.
        with pytest.warns(UserWarning, match="did not converge"):
            g = Chebtech1.from_function(jnp.sin, min_samples=8, max_length=8)
        assert g.coeffs.shape == (8,)

    def test_logical_true(self):
        # pass(8): chebtech1(@(x) x > -2) - chebtech1(1) has normest < eps.
        f = Chebtech1.from_function(lambda x: x > -2)
        g = Chebtech1.from_function(lambda x: jnp.ones_like(x))
        assert _normest(f - g) < EPS

    def test_logical_false(self):
        # pass(9): chebtech1(@(x) x < -2) - chebtech1(0) has normest < eps.
        f = Chebtech1.from_function(lambda x: x < -2)
        g = Chebtech1.from_function(lambda x: jnp.zeros_like(x))
        assert _normest(f - g) < EPS
