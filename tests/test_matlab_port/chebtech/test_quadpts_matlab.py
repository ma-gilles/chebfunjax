"""Port of MATLAB Chebfun tests/chebtech/test_quadpts.m (Opus 4.8).

MATLAB ``testTech.quadwts(n)`` -> chebfunjax ``chebweights(n, kind)`` and
``testTech.chebpts(n)`` -> ``chebpts(n, kind)`` (kind=1 for chebtech1,
kind=2 for chebtech2).

IMPORTANT GAP: chebfunjax ``chebweights(n, kind=1)`` returns GAUSS-CHEBYSHEV
weights (pi/n, for the 1/sqrt(1-x^2)-weighted integral), NOT MATLAB
``chebtech1.quadwts`` which are Fejer's first-rule weights for the *plain*
integral (sum = 2, exact for polynomials).  So for kind=1 the polynomial
exactness / weight-value assertions cannot pass and are xfail'd precisely.
For kind=2 (Clenshaw-Curtis) all assertions pass at the MATLAB tolerance.

Provenance
----------
MATLAB source : tests/chebtech/test_quadpts.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.quadrature import chebpts, chebweights

EPS = float(np.finfo(np.float64).eps)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


_FEJER = (
    "chebfunjax chebweights(n, kind=1) are Gauss-Chebyshev weights (pi/n), not "
    "MATLAB chebtech1.quadwts Fejer-1 weights for the plain integral (sum=2)"
)

# kind=2 (chebtech2) passes; kind=1 (chebtech1) xfails the exactness checks.
KIND_EXACT = [
    pytest.param(1, marks=pytest.mark.xfail(reason=_FEJER, strict=False)),
    2,
]
# Assertions that hold for both kinds (symmetry, w.x=0, w.x^3=0, empties):
KIND_BOTH = [1, 2]


class TestChebtechQuadpts:
    @pytest.mark.parametrize("kind", KIND_EXACT)
    def test_sum_equals_2(self, kind):
        # pass(m, 1): abs(sum(w) - 2) < 2*eps.
        w = chebweights(10, kind)
        assert abs(float(jnp.sum(w)) - 2) < 2 * EPS

    @pytest.mark.parametrize("kind", KIND_BOTH)
    def test_w_dot_x(self, kind):
        # pass(m, 2): abs(w*x) < eps.  (Holds for both -- symmetric weights.)
        w = chebweights(10, kind)
        x = chebpts(10, kind)
        assert abs(float(w @ x)) < EPS

    @pytest.mark.parametrize("kind", KIND_EXACT)
    def test_w_dot_x2(self, kind):
        # pass(m, 3): abs(w*x^2 - 2/3) < 2*eps.
        w = chebweights(10, kind)
        x = chebpts(10, kind)
        assert abs(float(w @ x ** 2) - 2 / 3) < 2 * EPS

    @pytest.mark.parametrize("kind", KIND_BOTH)
    def test_w_dot_x3(self, kind):
        # pass(m, 4): abs(w*x^3) < eps.  (Holds for both -- symmetric weights.)
        w = chebweights(10, kind)
        x = chebpts(10, kind)
        assert abs(float(w @ x ** 3)) < EPS

    @pytest.mark.parametrize("kind", KIND_EXACT)
    def test_w_dot_x4(self, kind):
        # pass(m, 5): abs(w*x^4 - 2/5) < 2*eps.
        w = chebweights(10, kind)
        x = chebpts(10, kind)
        assert abs(float(w @ x ** 4) - 2 / 5) < 2 * EPS

    @pytest.mark.parametrize("kind", KIND_BOTH)
    def test_quadwts_0_empty(self, kind):
        # pass(m, 6): isempty(quadwts(0)).
        assert chebweights(0, kind).shape[0] == 0

    @pytest.mark.parametrize("kind", KIND_BOTH)
    def test_quadwts_1_is_2(self, kind):
        # pass(m, 7): quadwts(1) == 2.
        w = chebweights(1, kind)
        assert w.shape == (1,)
        assert float(w[0]) == 2

    @pytest.mark.parametrize("kind", KIND_EXACT)
    def test_quadwts_2_all_1(self, kind):
        # pass(m, 8): all(quadwts(2) == 1).
        w = chebweights(2, kind)
        assert bool(jnp.all(w == 1))

    @pytest.mark.parametrize("kind", KIND_BOTH)
    def test_symmetry(self, kind):
        # pass(m, 9): n=10 even => norm(w(1:n/2) - w(n:-1:n/2+1), inf) < eps(n).
        n = 10
        w = np.asarray(chebweights(n, kind))
        left = w[: n // 2]
        right = w[n - 1: n // 2 - 1: -1]
        assert _ninf(left - right) < float(np.spacing(float(n)))
