"""Port of MATLAB Chebfun tests/trigtech/test_isinf.m (Opus 4.8[1m]).

isinf(f) is True iff f has any infinite value.

Provenance
----------
MATLAB source : tests/trigtech/test_isinf.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.tech.trigtech import Trigtech


def _inf_values(n=11):
    y = jnp.ones(n, dtype=jnp.float64).at[3].set(jnp.inf)
    return y


class TestTrigtechIsinf:
    def test_scalar_inf(self):
        f = Trigtech.from_values(_inf_values())
        assert f.isinf()

    def test_array_inf(self):
        y = _inf_values()
        f = Trigtech.from_values(jnp.stack([y, y], axis=-1))
        assert f.isinf()

    def test_scalar_finite(self):
        f = Trigtech.from_function(lambda x: jnp.cos(jnp.pi * x))
        assert not f.isinf()

    def test_array_finite(self):
        f = Trigtech.from_function(lambda x: jnp.stack(
            [jnp.cos(jnp.pi * x), jnp.cos(jnp.pi * x)], axis=-1))
        assert not f.isinf()
