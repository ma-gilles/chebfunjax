"""Port of MATLAB Chebfun tests/chebop/test_intops.m (Fable 5).

MATLAB's free functions ``fred(K, u)`` / ``volt(K, u)`` map to
:func:`chebfunjax.operators.integral.fred` / ``volt``.

Provenance
----------
MATLAB source : tests/chebop/test_intops.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop
from chebfunjax.operators.integral import fred, volt

jax.config.update("jax_enable_x64", True)

TOL = 1e-10


def _n(f, d, n=33):
    xs = jnp.linspace(d[0] + 1e-9, d[1] - 1e-9, n)
    return float(jnp.max(jnp.abs(jnp.asarray(f(xs)))))


class TestChebopIntops:
    def test_all_matlab_assertions(self):
        # pass(1): u' + 2u + 5*cumsum(u) = 1, u(0) = 0
        d = (0.0, 5.0)
        N = Chebop(lambda x, u: u.diff() + 2.0 * u + 5.0 * u.cumsum(),
                   domain=d)
        N.lbc = 0.0
        u = N.solve(1.0)
        x = cj.chebfun(lambda t: t, domain=d)
        u_exact = 0.5 * (-x).exp() * (2.0 * x).sin()
        assert _n(u - u_exact, d) < TOL

        # pass(2): Fredholm: A = u + fred(K, u), K = sin(2*pi*(x-y))
        d = (0.0, 1.0)
        K = lambda x, y: jnp.sin(2 * jnp.pi * (x - y))
        A = Chebop(lambda x, u: u + fred(K, u), domain=d)
        x = cj.chebfun(lambda t: t, domain=d)
        u = x * x.exp()
        f = A * u
        u2 = A.solve(f)
        assert _n(u - u2, d) < TOL

        # pass(3)/(4): Volterra: A = u - volt(K, u), K = x*y
        d = (0.0, float(np.pi))
        K = lambda x, y: x * y
        A = Chebop(lambda x, u: u - volt(K, u), domain=d)
        x = cj.chebfun(lambda t: t, domain=d)
        f = x ** 2 * x.cos() + (1.0 - x) * x.sin()
        u = A.solve(f)
        assert _n(u - x.sin(), d) < TOL
        assert _n(A * u - f, d) < TOL

        # pass(6): u' + fred(K, u) = 1, u(0) = 0, K = exp(-(x-y))
        d = (-1.0, 1.0)
        K = lambda x, y: jnp.exp(-(x - y))
        A = Chebop(lambda x, u: u.diff() + fred(K, u), domain=d)
        A.lbc = 0.0
        u = A.solve(1.0)
        assert _n(A * u - 1.0, d) < TOL

        # pass(7): u' + sum(u) = 0, u(0) = 1 -> u = 1 - 2x/3
        d = (0.0, 1.0)
        L = Chebop(lambda u: u.diff() + u.sum(), domain=d)
        L.lbc = 1.0
        u = L.solve(0.0)
        x = cj.chebfun(lambda t: t, domain=d)
        assert _n(u - (1.0 - 2.0 * x / 3.0), d) < TOL
