"""Port of MATLAB Chebfun tests/chebtech/test_bary.m (Fable 5).

Tests the barycentric interpolation formula on Chebyshev grids.  The MATLAB
test loops ``for n = 1:2`` over ``{chebtech1(), chebtech2()}``; the class's
own ``testclass.bary(y, fx)`` uses that class's Chebyshev points and weights.
chebfunjax's ``bary`` is a free function, so we pass the nodes and barycentric
weights explicitly: ``bary(y, f(xk), xk, bary_weights(xk))`` with ``xk`` the
Chebyshev points of the appropriate kind (1st kind for Chebtech1, 2nd kind for
Chebtech2).

Both MATLAB assertions are ported at MATLAB's tolerance
(``20*pref.chebfuneps``); ``bary`` handles ``(n, cols)`` data matrices, so
the array-valued ``[fx fx]`` case (pass 2) ports directly.  No gaps.

Provenance
----------
MATLAB source : tests/chebtech/test_bary.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.utils.interpolation import bary, bary_weights
from chebfunjax.utils.quadrature import chebpts

EPS = float(np.finfo(np.float64).eps)
# MATLAB tol = 20*pref.chebfuneps == 20*eps.
TOL = 20 * EPS

# (Tech, Chebyshev-point kind).
TECHS = [(Chebtech1, 1), (Chebtech2, 2)]


class TestChebtechBary:
    @pytest.mark.parametrize("Tech,kind", TECHS)
    def test_second_kind_formula(self, Tech, kind):
        # pass(n, 1): interpolate sin at k Chebyshev nodes, evaluate at m points.
        k = 14
        m = 10
        xk = chebpts(k, kind)
        y = jnp.asarray(np.linspace(-1.0, 1.0, m))
        fx = jnp.sin(xk)
        fy = jnp.sin(y)
        approx = bary(y, fx, xk, bary_weights(xk))
        assert float(jnp.linalg.norm(approx - fy)) < TOL

    @pytest.mark.parametrize("Tech,kind", TECHS)
    def test_array_valued(self, Tech, kind):
        # pass(n, 2): bary(y, [fx fx]) == [fy fy]
        k = 14
        m = 10
        xk = chebpts(k, kind)
        y = jnp.asarray(np.linspace(-1.0, 1.0, m))
        fx = jnp.sin(xk)
        fy = jnp.sin(y)
        approx = bary(y, jnp.stack([fx, fx], axis=-1), xk,
                      bary_weights(xk))
        exact = jnp.stack([fy, fy], axis=-1)
        assert float(jnp.linalg.norm(approx - exact)) < TOL
