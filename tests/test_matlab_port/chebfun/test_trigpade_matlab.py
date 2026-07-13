"""Port of MATLAB Chebfun tests/chebfun/test_trigpade.m (Fable 5).

FIXED: trigpade (Fourier-Pade via one-sided Laurent-Pade blocks)
added in the Fable 5 audit (Big-Three trig-rational directive).

Provenance
----------
MATLAB source : tests/chebfun/test_trigpade.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

TOL = 1e-12
RNG = np.random.default_rng(0)
TT = jnp.asarray(-1 + 2 * RNG.random(100))


class TestChebfunTrigpade:
    def test_sin_and_decomposition(self):
        f = cj.chebfun(lambda t: jnp.sin(np.pi * t), trig=True)
        p, q, r, s, t, u, v = cj.trigpade(f, 1, 0)
        # pass(4)-(5): p/q == f and the handle matches
        assert float(jnp.max(jnp.abs(p(TT) / q(TT) - f(TT)))) < TOL
        assert float(jnp.max(jnp.abs(f(TT) - r(TT)))) < TOL
        # pass(9): p/q == s/t + u/v
        assert float(jnp.max(jnp.abs(
            p(TT) / q(TT)
            - (s(TT) / t(TT) + u(TT) / v(TT))))) < TOL

    def test_genuine_rational(self):
        g = cj.chebfun(
            lambda t: 1.0 / (1.0 - 0.5 * jnp.cos(np.pi * t)),
            trig=True)
        _, _, r = cj.trigpade(g, 2, 2)[:3]
        assert float(jnp.max(jnp.abs(g(TT) - r(TT)))) < TOL
