"""Port of MATLAB Chebfun tests/chebtech2/test_alias.m (Opus 4.8).

chebfunjax has NO ``chebtech.alias`` function (coefficient aliasing /
folding onto a shorter Chebyshev grid).  Every assertion in this file is
therefore xfailed with that precise reason.  The test bodies build the
MATLAB-expected result and attempt ``Chebtech2.alias(...)`` so that, if an
``alias`` implementation is ever added, these turn into xpasses and get
noticed.  (The matrix cases additionally need array-valued techs, which
chebfunjax also lacks.)

Provenance
----------
MATLAB source : tests/chebtech2/test_alias.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech2, _clenshaw
from chebfunjax.utils.quadrature import chebpts

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS

_ALIAS_REASON = "chebfunjax lacks chebtech.alias (coefficient aliasing/folding)"


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestChebtech2Alias:
    @pytest.mark.xfail(reason=_ALIAS_REASON, strict=False)
    def test_padding(self):
        c0 = jnp.arange(10.0, 0.0, -1.0)
        c1 = Chebtech2.alias(c0, 11)
        assert _ninf(jnp.concatenate([c0, jnp.zeros(1)]) - c1) == 0

    @pytest.mark.xfail(reason=_ALIAS_REASON, strict=False)
    def test_aliasing_to_9(self):
        c0 = jnp.arange(10.0, 0.0, -1.0)
        c2 = Chebtech2.alias(c0, 9)
        expected = jnp.asarray([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 4.0, 2.0])
        assert _ninf(expected - c2) == 0

    @pytest.mark.xfail(reason=_ALIAS_REASON, strict=False)
    def test_aliasing_to_3(self):
        c0 = jnp.arange(10.0, 0.0, -1.0)
        c3 = Chebtech2.alias(c0, 3)
        assert _ninf(jnp.asarray([18.0, 25.0, 12.0]) - c3) == 0

    @pytest.mark.xfail(reason=_ALIAS_REASON, strict=False)
    def test_matches_eval_on_grid_9(self):
        c0 = jnp.arange(10.0, 0.0, -1.0)
        c2 = Chebtech2.alias(c0, 9)
        ref = Chebtech2.vals2coeffs(_clenshaw(c0, chebpts(9, kind=2)))
        assert _ninf(ref - c2) < TOL

    @pytest.mark.xfail(reason=_ALIAS_REASON, strict=False)
    def test_matches_eval_on_grid_3(self):
        c0 = jnp.arange(10.0, 0.0, -1.0)
        c3 = Chebtech2.alias(c0, 3)
        ref = Chebtech2.vals2coeffs(_clenshaw(c0, chebpts(3, kind=2)))
        assert _ninf(ref - c3) < TOL

    @pytest.mark.xfail(reason=_ALIAS_REASON, strict=False)
    def test_matrix_padding(self):
        c0 = jnp.arange(10.0, 0.0, -1.0)
        cc = jnp.stack([c0, c0[::-1]], axis=1)
        c1 = Chebtech2.alias(cc, 11)
        assert _ninf(jnp.concatenate([cc, jnp.zeros((1, 2))], axis=0) - c1) == 0

    @pytest.mark.xfail(reason=_ALIAS_REASON, strict=False)
    def test_matrix_aliasing_to_9(self):
        c0 = jnp.arange(10.0, 0.0, -1.0)
        cc = jnp.stack([c0, c0[::-1]], axis=1)
        c2 = Chebtech2.alias(cc, 9)
        col1 = jnp.asarray([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 4.0, 2.0])
        col2 = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 18.0, 9.0])
        expected = jnp.stack([col1, col2], axis=1)
        assert _ninf(expected - c2) == 0

    @pytest.mark.xfail(reason=_ALIAS_REASON, strict=False)
    def test_matrix_aliasing_to_3(self):
        c0 = jnp.arange(10.0, 0.0, -1.0)
        cc = jnp.stack([c0, c0[::-1]], axis=1)
        c3 = Chebtech2.alias(cc, 3)
        expected = jnp.asarray([[18.0, 15.0], [25.0, 30.0], [12.0, 10.0]])
        assert _ninf(expected - c3) == 0

    @pytest.mark.xfail(reason=_ALIAS_REASON, strict=False)
    def test_matrix_matches_eval_on_grid_9(self):
        c0 = jnp.arange(10.0, 0.0, -1.0)
        cc = jnp.stack([c0, c0[::-1]], axis=1)
        c2 = Chebtech2.alias(cc, 9)
        ref = Chebtech2.vals2coeffs(_clenshaw(cc, chebpts(9, kind=2)))
        assert _ninf(ref - c2) < TOL

    @pytest.mark.xfail(reason=_ALIAS_REASON, strict=False)
    def test_matrix_matches_eval_on_grid_3(self):
        c0 = jnp.arange(10.0, 0.0, -1.0)
        cc = jnp.stack([c0, c0[::-1]], axis=1)
        c3 = Chebtech2.alias(cc, 3)
        ref = Chebtech2.vals2coeffs(_clenshaw(cc, chebpts(3, kind=2)))
        assert _ninf(ref - c3) < TOL

    @pytest.mark.xfail(reason=_ALIAS_REASON, strict=False)
    def test_large_tail(self):
        c0 = 1.0 / jnp.arange(1.0, 1001.0) ** 5
        n = 17
        c1 = Chebtech2.alias(c0, n)
        v0 = Chebtech2.coeffs2vals(c0)
        v2 = Chebtech2.bary(chebpts(n, kind=2), v0)
        c2 = Chebtech2.vals2coeffs(v2)
        assert _ninf(c1 - c2) < n * TOL
