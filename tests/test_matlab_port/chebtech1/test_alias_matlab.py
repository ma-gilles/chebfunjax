"""Port of MATLAB Chebfun tests/chebtech1/test_alias.m (Opus 4.8).

chebfunjax has NO ``chebtech.alias`` function (coefficient aliasing /
folding onto a shorter Chebyshev grid).  Every assertion in this file is
therefore xfailed with that precise reason.  The test bodies build the
MATLAB-expected result and attempt ``Chebtech1.alias(...)`` so that, if an
``alias`` implementation is ever added, these turn into xpasses and get
noticed.  (The matrix cases additionally need array-valued techs, which
chebfunjax also lacks.)

Provenance
----------
MATLAB source : tests/chebtech1/test_alias.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, _clenshaw
from chebfunjax.utils.quadrature import chebpts

EPS = float(np.finfo(np.float64).eps)
TOL = 2e2 * EPS

_ALIAS_REASON = "chebfunjax lacks chebtech.alias (coefficient aliasing/folding)"


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestChebtech1Alias:
    @pytest.mark.xfail(reason=_ALIAS_REASON, strict=False)
    def test_padding(self):
        c0 = jnp.arange(10.0, 0.0, -1.0)
        c1 = Chebtech1.alias(c0, 11)
        assert _ninf(jnp.concatenate([c0, jnp.zeros(1)]) - c1) == 0

    @pytest.mark.xfail(reason=_ALIAS_REASON, strict=False)
    def test_aliasing_to_9(self):
        c0 = jnp.arange(10.0, 0.0, -1.0)
        c2 = Chebtech1.alias(c0, 9)
        assert _ninf(jnp.arange(10.0, 1.0, -1.0) - c2) == 0

    @pytest.mark.xfail(reason=_ALIAS_REASON, strict=False)
    def test_aliasing_to_3(self):
        c0 = jnp.arange(10.0, 0.0, -1.0)
        c3 = Chebtech1.alias(c0, 3)
        assert _ninf(jnp.asarray([6.0, 1.0, 0.0]) - c3) == 0

    @pytest.mark.xfail(reason=_ALIAS_REASON, strict=False)
    def test_matches_eval_on_grid_9(self):
        c0 = jnp.arange(10.0, 0.0, -1.0)
        c2 = Chebtech1.alias(c0, 9)
        ref = Chebtech1.vals2coeffs(_clenshaw(c0, chebpts(9, kind=1)))
        assert _ninf(ref - c2) < TOL

    @pytest.mark.xfail(reason=_ALIAS_REASON, strict=False)
    def test_matches_eval_on_grid_3(self):
        c0 = jnp.arange(10.0, 0.0, -1.0)
        c3 = Chebtech1.alias(c0, 3)
        ref = Chebtech1.vals2coeffs(_clenshaw(c0, chebpts(3, kind=1)))
        assert _ninf(ref - c3) < TOL

    @pytest.mark.xfail(reason=_ALIAS_REASON, strict=False)
    def test_matrix_padding(self):
        c0 = jnp.arange(10.0, 0.0, -1.0)
        cc = jnp.stack([c0, c0[::-1]], axis=1)
        c1 = Chebtech1.alias(cc, 11)
        assert _ninf(jnp.concatenate([cc, jnp.zeros((1, 2))], axis=0) - c1) == 0

    @pytest.mark.xfail(reason=_ALIAS_REASON, strict=False)
    def test_matrix_aliasing_to_9(self):
        c0 = jnp.arange(10.0, 0.0, -1.0)
        cc = jnp.stack([c0, c0[::-1]], axis=1)
        c2 = Chebtech1.alias(cc, 9)
        expected = jnp.stack(
            [jnp.arange(10.0, 1.0, -1.0), jnp.arange(1.0, 10.0)], axis=1
        )
        assert _ninf(expected - c2) == 0

    @pytest.mark.xfail(reason=_ALIAS_REASON, strict=False)
    def test_matrix_aliasing_to_3(self):
        c0 = jnp.arange(10.0, 0.0, -1.0)
        cc = jnp.stack([c0, c0[::-1]], axis=1)
        c3 = Chebtech1.alias(cc, 3)
        expected = jnp.asarray([[6.0, -6.0], [1.0, -12.0], [0.0, -11.0]])
        assert _ninf(expected - c3) == 0

    @pytest.mark.xfail(reason=_ALIAS_REASON, strict=False)
    def test_matrix_matches_eval_on_grid_9(self):
        c0 = jnp.arange(10.0, 0.0, -1.0)
        cc = jnp.stack([c0, c0[::-1]], axis=1)
        c2 = Chebtech1.alias(cc, 9)
        ref = Chebtech1.vals2coeffs(_clenshaw(cc, chebpts(9, kind=1)))
        assert _ninf(ref - c2) < TOL

    @pytest.mark.xfail(reason=_ALIAS_REASON, strict=False)
    def test_matrix_matches_eval_on_grid_3(self):
        c0 = jnp.arange(10.0, 0.0, -1.0)
        cc = jnp.stack([c0, c0[::-1]], axis=1)
        c3 = Chebtech1.alias(cc, 3)
        ref = Chebtech1.vals2coeffs(_clenshaw(cc, chebpts(3, kind=1)))
        assert _ninf(ref - c3) < TOL

    @pytest.mark.xfail(reason=_ALIAS_REASON, strict=False)
    def test_large_tail(self):
        c0 = 1.0 / jnp.arange(1000.0, 0.0, -1.0) ** 5
        n = 17
        c1 = Chebtech1.alias(c0, n)
        v0 = Chebtech1.coeffs2vals(c0)
        v2 = Chebtech1.bary(chebpts(n, kind=1), v0)
        c2 = Chebtech1.vals2coeffs(v2)
        assert _ninf(c1 - c2) < n * TOL
