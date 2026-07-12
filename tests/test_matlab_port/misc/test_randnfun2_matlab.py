"""Port of MATLAB Chebfun tests/misc/test_randnfun2.m (Fable 5).

FIXED: randnfun2 added in the Fable 5 audit.  MATLAB's rng-based
reproducibility maps to the numpy ``seed`` kwarg; the translation /
scale-shift invariance assertions (pass 3-4, 8-9) hold for the same
seed exactly as in MATLAB.

Provenance
----------
MATLAB source : tests/misc/test_randnfun2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

import chebfunjax as cj


class TestRandnfun2:
    def test_moments_and_invariance(self):
        f = cj.randnfun2(0.1, seed=0)
        # pass(1)-(2): unit pointwise variance, zero mean
        assert abs(float((f * f).mean2()) - 1) < 0.1
        assert abs(float(f.mean2())) < 0.1

        # pass(3): same seed on a translated domain -> shifted function
        g = cj.randnfun2(0.1, (2, 4, -1, 1), seed=0)
        assert abs(float(f(jnp.asarray(0.5), jnp.asarray(0.5)))
                   - float(g(jnp.asarray(3.5), jnp.asarray(0.5)))) \
            < 1e-12

        # pass(4): doubled lambda on a doubled domain -> same pattern
        h = cj.randnfun2(0.2, (-2, 2, 0, 4), seed=0)
        assert abs(float(f(jnp.asarray(0.5), jnp.asarray(0.5)))
                   - float(h(jnp.asarray(1.0), jnp.asarray(3.0)))) \
            < 1e-12

    def test_trig_variant(self):
        # pass(6)-(7)
        f = cj.randnfun2(0.1, seed=0, trig=True)
        assert abs(float((f * f).mean2()) - 1) < 0.1
        assert abs(float(f.mean2())) < 0.1

    def test_huge_lambda_nearly_constant(self):
        # pass(5): lambda >> domain -> derivative ~ 0
        f = cj.randnfun2(1e6, seed=0)
        d = float(jnp.max(jnp.abs(
            f.diff(1)(jnp.zeros(1), jnp.zeros(1)))))
        assert d < 1e-4
