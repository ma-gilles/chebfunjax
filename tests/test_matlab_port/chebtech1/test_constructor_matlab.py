"""Port of MATLAB Chebfun tests/chebtech1/test_constructor.m (Fable 5).

MATLAB's ``test_constructor`` exercises the non-user-facing ``populate()``.
chebfunjax's equivalent entry point is ``Chebtech1.from_function``, a single
adaptive constructor: the scalar and array-valued construction-accuracy
checks, the NaN/Inf error, the fixed-length construction and the
logical-valued construction all port at the MATLAB tolerances.  ``normest``
is reproduced as ``max(vscale)`` (a faithful equivalent of
@chebtech/normest.m, which is literally ``out = max(vscale(f))``).

Gap: MATLAB's ``pref.refinementFunction = 'resampling'`` (pass 3 and 4) has
no counterpart -- chebfunjax has a single adaptive refinement path,
equivalent to MATLAB's default 'nested', so those two assertions would only
re-run pass(1)/pass(2) against the same constructor.  They stay skipped.

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
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


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
        pytest.skip(
            "pass(3): chebfunjax has a single adaptive refinement path "
            "(equivalent to MATLAB's default pref.refinementFunction="
            "'nested'); there is no 'resampling' variant, so this assertion "
            "would only re-run test_scalar_sin_nested"
        )

    def test_array_sin_cos_exp_resampling(self):
        pytest.skip(
            "pass(4): chebfunjax has a single adaptive refinement path "
            "(equivalent to MATLAB's default pref.refinementFunction="
            "'nested'); there is no 'resampling' variant, so this assertion "
            "would only re-run test_array_sin_cos_exp_nested"
        )

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
        # pass(7): construction must not crash when pref.minSamples ==
        # pref.maxLength == 8, i.e. when the adaptive loop is pinned to a
        # single length.  chebfunjax spells that as the fixed-length option
        # from_function(..., n=8).
        g = Chebtech1.from_function(jnp.sin, n=8)
        assert g.coeffs.shape == (8,)

    def test_logical_true(self):
        # pass(8): chebtech1(@(x) x > -2) - chebtech1(1) has normest < eps.
        f = Chebtech1.from_function(lambda x: (x > -2).astype(jnp.float64))
        g = Chebtech1.from_function(lambda x: jnp.ones_like(x))
        assert _normest(f - g) < EPS

    def test_logical_false(self):
        # pass(9): chebtech1(@(x) x < -2) - chebtech1(0) has normest < eps.
        f = Chebtech1.from_function(lambda x: (x < -2).astype(jnp.float64))
        g = Chebtech1.from_function(lambda x: jnp.zeros_like(x))
        assert _normest(f - g) < EPS
