"""Port of MATLAB Chebfun tests/chebfun/test_kron.m (Fable 5).

FIXED (adapted): kron(f, g)(x, y) = f(x) g(y) added in the Fable 5
audit -- MATLAB's kron(f', g) row/column orderings map to the two
argument orders (chebfunjax has no transposed chebfuns).

Provenance
----------
MATLAB source : tests/chebfun/test_kron.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun2d.chebfun2 import Chebfun2

TOL = 1e-10


def _maxdiff(h, exact, dom):
    xs = jnp.asarray(np.linspace(dom[0], dom[1], 11))
    ys = jnp.asarray(np.linspace(dom[2], dom[3], 11))
    xx, yy = jnp.meshgrid(xs, ys, indexing="ij")
    return float(jnp.max(jnp.abs(h(xx, yy) - exact(xx, yy))))


class TestChebfunKron:
    def test_rank1_products(self):
        f = cj.chebfun(lambda x: x ** 2)
        g = cj.chebfun(jnp.sin)
        # pass(1): kron(f', g) = x^2 sin(y)
        h1 = cj.kron(f, g)
        assert isinstance(h1, Chebfun2)
        assert _maxdiff(h1, lambda X, Y: X ** 2 * jnp.sin(Y),
                        (-1, 1, -1, 1)) < TOL
        # pass(2): the other ordering = y^2 sin(x)
        h2 = cj.kron(g, f)
        assert _maxdiff(h2, lambda X, Y: jnp.sin(X) * Y ** 2,
                        (-1, 1, -1, 1)) < TOL

    def test_different_domains(self):
        d = (-2.0, float(np.pi), -float(np.pi), 2.0)
        f = cj.chebfun(lambda x: x ** 2, domain=d[:2])
        g = cj.chebfun(jnp.sin, domain=d[2:])
        h1 = cj.kron(f, g)
        assert _maxdiff(h1, lambda X, Y: X ** 2 * jnp.sin(Y), d) \
            < TOL
