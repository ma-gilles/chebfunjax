"""Port of MATLAB Chebfun tests/chebfun/test_vectorCheck.m (Fable 5).

MATLAB's vectorCheck broadcasts scalar / constant-row outputs, fixes a
transposed array-valued output and falls back to pointwise evaluation
for an operator that cannot take a vector (``quadgk``); the ``'vectorize'``
flag is ``vectorize=True``.

Provenance
----------
MATLAB source : tests/chebfun/test_vectorCheck.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from chebfunjax.chebfun1d.chebfun import chebfun

jax.config.update("jax_enable_x64", True)


def _n(f):
    # MATLAB norm of an array-valued chebfun: Frobenius over the columns.
    return float(jnp.linalg.norm(jnp.atleast_1d(jnp.asarray(f.norm(2)))))


def _normest(f):
    x = jnp.asarray(np.linspace(0.0, 10.0, 200))
    return float(np.max(np.abs(np.asarray(f(x)))))


class TestChebfunVectorCheck:
    def test_all_matlab_assertions(self):
        f = chebfun(lambda x: 1)
        g = chebfun(lambda x: 1 + 0 * x)
        assert _n(f - g) == 0                                       # pass(1)

        f = chebfun(lambda x: jnp.sin(x).T)
        g = chebfun(jnp.sin)
        assert _n(f - g) == 0                                       # pass(2)

        qg = lambda x: quad(np.cos, float(x), 2.0)[0]  # noqa: E731
        f = chebfun(qg)
        g = chebfun(qg, vectorize=True)
        assert _n(f - g) == 0                                       # pass(3)

        f = chebfun(lambda x: jnp.array([1.0, 2.0, 3.0]))
        g = chebfun(lambda x: jnp.stack([1 + 0 * x, 2 + 0 * x, 3 + 0 * x], axis=-1))
        assert _n(f - g) == 0                                       # pass(4)

        f = chebfun(lambda x: jnp.stack([x, x]), domain=(-1.0, 1.0))      # [x x].'
        g = chebfun(lambda x: jnp.stack([x, x], axis=-1), domain=(-1.0, 1.0))
        assert _n(f - g) == 0                                       # pass(5)

        f = chebfun(lambda x: jnp.stack([x, x]), domain=(-1.0, 0.0, 1.0))
        g = chebfun(lambda x: jnp.stack([x, x], axis=-1), domain=(-1.0, 0.0, 1.0))
        assert _n(f - g) == 0                                       # pass(6)

        f = chebfun(lambda x: jnp.stack([x, x, x]), domain=(-1.0, 1.0))
        g = chebfun(lambda x: jnp.stack([x, x, x], axis=-1), domain=(-1.0, 1.0))
        assert _n(f - g) == 0                                       # pass(7)

        f = chebfun(lambda x: jnp.stack([jnp.exp(-x ** 2), jnp.exp(-x ** 2)]),
                    domain=(0.0, jnp.inf))
        g = chebfun(lambda x: jnp.stack([jnp.exp(-x ** 2), jnp.exp(-x ** 2)], axis=-1),
                    domain=(0.0, jnp.inf))
        assert _normest(f - g) < 1e-10                              # pass(8)

        f = chebfun(lambda x: jnp.stack([jnp.sin(x), 1 + 0 * x], axis=-1))
        g = chebfun(lambda x: jnp.stack([jnp.sin(x), 1 + 0 * x], axis=-1))
        assert _n(f - g) == 0                                       # pass(9)

        f = chebfun(lambda x: jnp.array([1.0, 1.0]))
        g = chebfun(lambda x: jnp.stack([1 + 0 * x, 1 + 0 * x], axis=-1))
        assert _n(f - g) == 0                                       # pass(10)

        f = chebfun(lambda x: x / jnp.cos(x))
        g = chebfun(lambda x: x / jnp.cos(x))
        assert _n(f - g) == 0                                       # pass(11)
