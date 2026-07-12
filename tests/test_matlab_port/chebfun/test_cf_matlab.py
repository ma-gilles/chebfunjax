"""Port of MATLAB Chebfun tests/chebfun/test_cf.m (Fable 5).

FIXED: cf (Caratheodory-Fejer) added in the Fable 5 audit
(polynomial + rational branches, faithful port of cf.m incl.
getBlock and the FFT Laurent machinery).  Quasimatrix cases skipped.

Provenance
----------
MATLAB source : tests/chebfun/test_cf.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestChebfunCf:
    def test_polynomial_cf_lambda(self):
        # pass(1)-(2): lam ~ 0.045017 for exp, degree 2
        f = cj.chebfun(jnp.exp)
        _, _, _, lam = cj.cf(f, 2)
        assert abs(lam - 0.045017) < 1e-4

        f2 = cj.chebfun(lambda x: jnp.exp(-1 + x / 2), domain=(0, 4))
        _, _, _, lam2 = cj.cf(f2, 2)
        assert abs(lam2 - 0.045017) < 1e-4

    def test_rational_cases(self):
        # pass(3): cf(cos, 1, 1)
        f = cj.chebfun(jnp.cos)
        p, _, _, _ = cj.cf(f, 1, 1)
        assert abs(float(p(jnp.asarray(0.3))) - 0.77015046914) < 1e-4

        # pass(4): |x| with (4, 4), M = 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fa = cj.chebfun(lambda x: jnp.abs(x), splitting=True)
            _, _, r, _ = cj.cf(fa, 4, 4, 100)
        xx = jnp.asarray(np.linspace(-1, 1, 401))
        l2 = float(jnp.sqrt(jnp.trapezoid(
            (fa(xx) - r(xx)) ** 2, xx)))
        assert l2 < 0.05

        # pass(5): exp(exp(x)), (0, 10)
        fe = cj.chebfun(lambda x: jnp.exp(jnp.exp(x)))
        _, _, r5, _ = cj.cf(fe, 0, 10)
        xs = jnp.asarray(np.linspace(-1, 1, 17))
        assert float(jnp.max(jnp.abs(fe(xs) - r5(xs)))) < 1e-4

        # pass(6): exp on [2, 6], (5, 5)
        f6 = cj.chebfun(jnp.exp, domain=(2, 6))
        _, _, r6, _ = cj.cf(f6, 5, 5)
        xs6 = jnp.asarray(np.linspace(2, 6, 100))
        assert float(jnp.max(jnp.abs(f6(xs6) - r6(xs6)))) < 1e-6
