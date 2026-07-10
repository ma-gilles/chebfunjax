"""Port of MATLAB Chebfun tests/chebtech1/test_constructor.m (Opus 4.8).

MATLAB's ``test_constructor`` exercises the non-user-facing ``populate()``
with ``pref.refinementFunction`` ('nested' / 'resampling'), NaN/Inf error
handling, ``minSamples``/``maxLength`` prefs and logical-valued
construction.  chebfunjax has none of that machinery: ``from_function`` is
a single adaptive constructor with no prefs.

The one faithfully portable assertion is the scalar ``sin`` construction
accuracy check (nested / default refinement), which is reproduced exactly
at the MATLAB tolerance ``10*vscale(g)*eps``.  Everything else is skipped
with a precise reason.

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


class TestChebtech1Constructor:
    def test_scalar_sin_nested(self):
        # MATLAB pass(1): populate with refinementFunction='nested', scalar sin.
        g = Chebtech1.from_function(jnp.sin)
        x = chebpts(len(g.coeffs), kind=1)
        values = Chebtech1.coeffs2vals(g.coeffs)
        assert _ninf(jnp.sin(x) - values) < 10 * g.vscale * EPS

    def test_array_sin_cos_exp_nested(self):
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix techs"
        )

    def test_scalar_sin_resampling(self):
        pytest.skip(
            "chebfunjax has no pref.refinementFunction='resampling' "
            "(single adaptive construction path)"
        )

    def test_array_sin_cos_exp_resampling(self):
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

    def test_minsamples_equals_maxlength(self):
        pytest.skip(
            "chebfunjax has no pref.minSamples/pref.maxLength construction options"
        )

    def test_logical_true(self):
        pytest.skip(
            "chebfunjax has no logical-valued construction / normest(); "
            "cannot port chebtech1(@(x) x > -2)"
        )

    def test_logical_false(self):
        pytest.skip(
            "chebfunjax has no logical-valued construction / normest(); "
            "cannot port chebtech1(@(x) x < -2)"
        )
