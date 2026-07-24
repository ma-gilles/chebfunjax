"""Port of MATLAB Chebfun tests/trigtech/test_isfinite.m (Opus 4.8[1m]).

isfinite(f) is True iff f is everywhere finite.

Provenance
----------
MATLAB source : tests/trigtech/test_isfinite.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.tech.trigtech import Trigtech, trigpts


def _inf_values(n=11):
    return jnp.ones(n, dtype=jnp.float64).at[3].set(jnp.inf)


class TestTrigtechIsfinite:
    def test_scalar_inf(self):
        f = Trigtech.from_values(_inf_values())
        assert not f.isfinite()

    def test_array_inf(self):
        y = _inf_values()
        f = Trigtech.from_values(jnp.stack([y, y], axis=-1))
        assert not f.isfinite()

    def test_scalar_finite(self):
        # A plain finite tech (MATLAB uses @(x) x; any finite sample works).
        x = trigpts(17)
        f = Trigtech.from_values(x)
        assert f.isfinite()

    def test_array_finite(self):
        x = trigpts(17)
        f = Trigtech.from_values(jnp.stack([x, x], axis=-1))
        assert f.isfinite()
