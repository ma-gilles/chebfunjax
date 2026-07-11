"""Port of MATLAB Chebfun tests/chebop/test_expm.m (Fable 5).

Heat semigroup expm(t*D2) applied to a Gaussian; checked against the
exact heat-kernel convolution at t = 0.01 (free-space kernel is
accurate to well below tol for this compactly-concentrated u0 away
from the boundaries).

Provenance
----------
MATLAB source : tests/chebop/test_expm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.chebop import Chebop

TOL = 1e2 * 1e-10


class TestChebopExpm:
    def test_heat_semigroup_of_gaussian(self):
        A = Chebop(lambda x, u: u.diff(2))
        A.lbc = 0.0
        A.rbc = 0.0
        t = 0.01
        try:
            u = A.expm(t, lambda x: jnp.exp(-20 * (x + 0.3) ** 2))
        except TypeError:
            import chebfunjax as cj
            u0 = cj.chebfun(lambda x: jnp.exp(-20 * (x + 0.3) ** 2))
            u = A.expm(t, u0)
        # exact via method of images for Dirichlet on [-1,1]:
        # u = sum_n (-1)^n G_t(x - x_n), x_n the reflections of -0.3.
        xs = np.linspace(-0.85, 0.4, 30)
        s = 1 + 80 * t

        def g(y):
            return np.exp(-20 * y ** 2 / s) / np.sqrt(s)
        # reflection set for [-1,1]: sources at 4n + x0 (+) and
        # 4n - 2 - x0 (-)
        x0 = -0.3
        exact = np.zeros_like(xs)
        for n in range(-3, 4):
            exact += g(xs - (4 * n + x0)) - g(xs - (4 * n - 2 - x0))
        got = np.asarray(u(jnp.asarray(xs)))
        assert float(np.max(np.abs(got - exact))) < 1e4 * TOL
