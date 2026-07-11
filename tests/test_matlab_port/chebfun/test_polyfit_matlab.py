"""Port of MATLAB Chebfun tests/chebfun/test_polyfit.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_polyfit.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-0.97, 0.97, 200))


class TestChebfunPolyfit:
    def test_degree5_fit_of_quartic_is_exact(self):
        F = cj.chebfun(lambda x: x ** 2 + 3 * x ** 4)
        p = F.polyfit(5)
        err = jnp.abs(p(X) - F(X))
        assert float(jnp.max(err)) < 100 * EPS * p.vscale

    def test_piecewise_quartic_degree6(self):
        F = cj.chebfun(lambda x: x ** 2 + 3 * x ** 4,
                       domain=[-1.0, -0.5, 0.0, 0.5, 1.0])
        p = F.polyfit(6)
        err = jnp.abs(p(X) - F(X))
        assert float(jnp.max(err)) < 100 * EPS * p.vscale

    def test_lower_degree_is_least_squares(self):
        # fitting x^4-ish with degree 2: L2 projection onto P_2
        F = cj.chebfun(lambda x: x ** 4)
        p = F.polyfit(2)
        # exact Legendre projection: x^4 -> (3/35) + (6/7)(x^2-1/3)...
        # closed form: p*(x) = (6/7)x^2 - 3/35
        exact = (6.0 / 7.0) * X ** 2 - 3.0 / 35.0
        err = jnp.abs(p(X) - exact)
        assert float(jnp.max(err)) < 1e4 * EPS
