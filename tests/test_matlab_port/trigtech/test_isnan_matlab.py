"""Port of MATLAB Chebfun tests/trigtech/test_isnan.m (Opus 4.8[1m]).

isnan(f) is True iff f has any NaN value.  In the coeffs-only model a
genuine NaN value produces NaN (but not Inf) Fourier coefficients.

Provenance
----------
MATLAB source : tests/trigtech/test_isnan.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.tech.trigtech import Trigtech, trigpts


class TestTrigtechIsnan:
    def test_scalar_finite(self):
        f = Trigtech.from_function(lambda x: jnp.cos(jnp.pi * x))
        assert not f.isnan()

    def test_array_finite(self):
        f = Trigtech.from_function(lambda x: jnp.stack(
            [jnp.cos(jnp.pi * x), jnp.cos(jnp.pi * x) ** 2], axis=-1))
        assert not f.isnan()

    def test_constructed_nan(self):
        # MATLAB make(NaN): a single-value NaN tech.
        f = Trigtech.from_values(jnp.array([jnp.nan], dtype=jnp.float64))
        assert f.isnan()

    def test_scalar_nan_function(self):
        # cos(pi x) + NaN sampled on a grid -> a NaN tech.
        n = 17
        x = trigpts(n)
        f = Trigtech.from_values(jnp.cos(jnp.pi * x) + jnp.nan)
        assert f.isnan()

    def test_array_nan_function(self):
        n = 17
        x = trigpts(n)
        vals = jnp.stack([jnp.cos(jnp.pi * x) + jnp.nan, jnp.cos(jnp.pi * x)],
                         axis=-1)
        f = Trigtech.from_values(vals)
        assert f.isnan()
