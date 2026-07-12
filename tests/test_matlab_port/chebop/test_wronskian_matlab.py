"""Port of MATLAB Chebfun tests/chebop/test_wronskian.m (Fable 5).

FIXED: wronskian added in the Fable 5 audit (derivative-matrix
determinant; the chebop first argument is accepted and ignored, as
it only fixes n in MATLAB).  The chebmatrix input style is skipped
(no chebmatrix class).

Provenance
----------
MATLAB source : tests/chebop/test_wronskian.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

TOL = 1e-11


class TestChebopWronskian:
    def test_cos_sin(self):
        # pass(1)-(2): W(cos, sin) = 1 on [-pi, pi]
        c = cj.chebfun(jnp.cos, domain=(-np.pi, np.pi))
        s = cj.chebfun(jnp.sin, domain=(-np.pi, np.pi))
        w = cj.wronskian(c, s)
        xs = jnp.asarray(np.linspace(-0.99 * np.pi, 0.99 * np.pi, 60))
        assert float(jnp.max(jnp.abs(w(xs) - 1))) < TOL

    def test_exponentials(self):
        # pass(4): W(e^-5x, e^x) = 6 e^-4x on [0, 5]
        f = cj.chebfun(lambda x: jnp.exp(-5 * x), domain=(0, 5))
        g = cj.chebfun(jnp.exp, domain=(0, 5))
        w = cj.wronskian(f, g)
        xs = jnp.asarray(np.linspace(0.01, 4.99, 60))
        assert float(jnp.max(jnp.abs(
            w(xs) - 6 * jnp.exp(-4 * xs)))) < TOL
