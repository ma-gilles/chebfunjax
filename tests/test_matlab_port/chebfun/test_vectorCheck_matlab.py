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

# Retained test-only callback adapter for native quadgk. This test does not
# port or qualify MATLAB quadgk; production construction kernels remain JAX.
from scipy.integrate import quad

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)


def _n(f):
    # MATLAB norm of an array-valued chebfun: Frobenius over the columns.
    return float(jnp.linalg.norm(jnp.atleast_1d(jnp.asarray(f.norm(2)))))


def _normest(f):
    return float(f.normest())


def _column_mrdivide_cos(x):
    # The two-point vectorCheck probe has nonzero real column divisor B.
    # A/B = A*B'/(B'*B) is the exact rank-one source shape adapter. Probe
    # values are discarded by the square-output scalar-column retry.
    x = jnp.asarray(x)
    divisor = jnp.cos(x)
    if x.ndim == 0:
        return x / divisor
    return x[:, None] * divisor[None, :] / jnp.sum(divisor * divisor)


class TestChebfunVectorCheck:
    def test_all_matlab_assertions(self):
        # Explicit source 2D shapes preserve MATLAB row/column orientation
        # at scalar probes (slots2,5,6,7,8) and constant-row slot10.
        # Source expressions restored; slot3 retains only its explicitly
        # labelled quadrature callback adapter. No sample-grid norm surrogate.
        pref = ChebfunPref()
        f = chebfun(lambda x: 1, pref=pref)
        g = chebfun(lambda x: 1 + 0 * x, pref=pref)
        assert _n(f - g) == 0                                       # pass(1)

        f = chebfun(lambda x: jnp.atleast_2d(jnp.sin(x)), pref=pref)
        g = chebfun(jnp.sin, pref=pref)
        assert _n(f - g) == 0                                       # pass(2), first assignment
        f = chebfun(lambda x: jnp.atleast_2d(jnp.sin(x)))
        assert _n(f - chebfun(jnp.sin)) == 0                         # pass(2), source reassignment

        qg = lambda x: quad(np.cos, float(x), 2.0)[0]  # noqa: E731
        f = chebfun(qg)
        g = chebfun(qg, vectorize=True)
        assert _n(f - g) == 0                                       # pass(3)

        f = chebfun(lambda x: jnp.array([1.0, 2.0, 3.0]))
        g = chebfun(jnp.asarray([[1., 2., 3.]]), pref=pref)
        assert _n(f - g) == 0                                       # pass(4)

        f = chebfun(lambda x: jnp.stack([jnp.atleast_1d(x), jnp.atleast_1d(x)]), domain=(-1.0, 1.0))      # [x x].'
        g = chebfun(lambda x: jnp.stack([x, x], axis=-1), domain=(-1.0, 1.0))
        assert _n(f - g) == 0                                       # pass(5)

        f = chebfun(lambda x: jnp.stack([jnp.atleast_1d(x), jnp.atleast_1d(x)]), domain=(-1.0, 0.0, 1.0))
        g = chebfun(lambda x: jnp.stack([x, x], axis=-1), domain=(-1.0, 0.0, 1.0))
        assert _n(f - g) == 0                                       # pass(6)

        f = chebfun(lambda x: jnp.stack([jnp.atleast_1d(x)] * 3), domain=(-1.0, 1.0))
        g = chebfun(lambda x: jnp.stack([x, x, x], axis=-1), domain=(-1.0, 1.0))
        assert _n(f - g) == 0                                       # pass(7)

        f = chebfun(lambda x: jnp.stack([jnp.atleast_1d(jnp.exp(-x ** 2))] * 2),
                    domain=(0.0, jnp.inf))
        g = chebfun(lambda x: jnp.stack([jnp.exp(-x ** 2), jnp.exp(-x ** 2)], axis=-1),
                    domain=(0.0, jnp.inf))
        assert _normest(f - g) < 1e-10                              # pass(8)

        # Native [sin(x) 1] fails horizontal concatenation for vector x,
        # requiring vec's scalar callback route; g is already vectorized.
        f = chebfun(lambda x: jnp.concatenate((jnp.atleast_1d(jnp.sin(x))[:, None],
                                               jnp.ones((1, 1))), axis=1), pref=pref)
        g = chebfun(lambda x: jnp.stack([jnp.sin(x), 1 + 0 * x], axis=-1), pref=pref)
        assert _n(f - g) == 0                                       # pass(9)

        f = chebfun(lambda x: jnp.array([[1.0, 1.0]]))
        g = chebfun(jnp.asarray([[1., 1.]]), pref=pref)
        assert _n(f - g) == 0                                       # pass(10)

        f = chebfun(_column_mrdivide_cos, pref=pref)
        g = chebfun(lambda x: x / jnp.cos(x), pref=pref)
        assert _n(f - g) == 0                                       # pass(11)
