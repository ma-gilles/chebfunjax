"""Port of MATLAB Chebfun tests/chebfun2/test_sum.m (Fable 5).

chebfunjax sum(dim) returns a rank-compressed Chebfun2 (constant in the
integrated variable) rather than a 1-D chebfun; assertions compare
values of the free variable, which is what MATLAB's norms check.

Provenance
----------
MATLAB source : tests/chebfun2/test_sum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS
XS = jnp.asarray(np.linspace(-0.95, 0.95, 30))


class TestChebfun2Sum:
    def test_integral2_polynomial(self):
        f = Chebfun2.from_function(lambda x, y: x ** 2 + 4 * y,
                                   domain=(11.0, 14.0, 7.0, 10.0))
        assert abs(float(f.sum2()) - 1719.0) < 30 * TOL * 1719

    def test_sum_over_y_of_x(self):
        # MATLAB: sum(f) integrates over y -> 2x as function of x.
        f = Chebfun2.from_function(lambda x, y: x)
        s = f.sum(dim=1)
        vals = jnp.asarray([s(jnp.asarray(float(x)), jnp.asarray(0.0))
                            for x in XS])
        assert float(jnp.max(jnp.abs(vals - 2 * XS))) < TOL

    def test_sum_over_x_of_x_is_zero(self):
        f = Chebfun2.from_function(lambda x, y: x)
        s = f.sum(dim=2)
        vals = jnp.asarray([s(jnp.asarray(0.0), jnp.asarray(float(y)))
                            for y in XS])
        assert float(jnp.max(jnp.abs(vals))) < 10 * EPS

    def test_sum_over_y_of_y_stretched(self):
        # f = y on [0 1]x[-pi pi]: sum over y = 0; sum over x = y.
        f = Chebfun2.from_function(lambda x, y: y,
                                   domain=(0.0, 1.0, -np.pi, np.pi))
        s1 = f.sum(dim=1)
        vals = jnp.asarray([s1(jnp.asarray(0.5), jnp.asarray(0.0))])
        assert float(jnp.max(jnp.abs(vals))) < 10 * EPS * np.pi
        s2 = f.sum(dim=2)
        ys = jnp.asarray(np.linspace(-3.0, 3.0, 20))
        vals2 = jnp.asarray([s2(jnp.asarray(0.5), jnp.asarray(float(y)))
                             for y in ys])
        assert float(jnp.max(jnp.abs(vals2 - ys))) < TOL * np.pi
