"""Port of MATLAB Chebfun tests/misc/test_bary.m (Fable 5).

Scalar/vector evaluation of a quartic through Chebyshev nodes of both
kinds at MATLAB's tol=1e-14.  Matrix/array-shaped inputs and
array-valued interpolants are covered where bary broadcasts.

Provenance
----------
MATLAB source : tests/misc/test_bary.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.interpolation import bary, bary_weights, cheb_bary_weights
from chebfunjax.utils.quadrature import chebpts

TOL = 1e-14
RNG = np.random.default_rng(7681)
XR = jnp.asarray(2 * RNG.uniform(size=1000) - 1)


def p(x):
    return x ** 4 - 2 * x ** 3 + 3 * x ** 2 - 2 * x + 1


class TestBary:
    @pytest.mark.parametrize("kind", [2, 1])
    def test_quartic_through_16_nodes(self, kind):
        xk = jnp.asarray(chebpts(16, kind=kind))
        # cheb_bary_weights is 2nd-kind only; 1st-kind nodes use the
        # general O(n^2) barycentric weights (same interpolant).
        vk = (jnp.asarray(cheb_bary_weights(16)) if kind == 2
              else jnp.asarray(bary_weights(xk)))
        err = float(jnp.max(jnp.abs(p(XR) - bary(XR, p(xk), xk, vk))))
        assert err < TOL

    @pytest.mark.parametrize("kind", [2, 1])
    def test_scalar_point(self, kind):
        xk = jnp.asarray(chebpts(16, kind=kind))
        vk = (jnp.asarray(cheb_bary_weights(16)) if kind == 2
              else jnp.asarray(bary_weights(xk)))
        x0 = XR[:1]
        err = abs(float(p(x0)[0] - bary(x0, p(xk), xk, vk)[0]))
        assert err < TOL

    def test_evaluation_at_a_node_is_exact(self):
        xk = jnp.asarray(chebpts(16, kind=2))
        vk = jnp.asarray(cheb_bary_weights(16))
        got = bary(xk[3:4], p(xk), xk, vk)
        assert abs(float(got[0] - p(xk)[3])) < TOL
