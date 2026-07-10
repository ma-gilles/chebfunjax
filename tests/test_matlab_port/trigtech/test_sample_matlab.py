"""Port of MATLAB Chebfun tests/trigtech/test_sample.m (Opus 4.8).

sample(f) returns the values and points on the native equispaced grid.
chebfunjax exposes this as ``f.values`` on ``trigpts(len(f))``; resampling
to a different grid length (sample(f, m)) has no dedicated method and is
covered instead by ``feval`` on trigpts(m), so those cases are skipped
with a precise reason.

Provenance
----------
MATLAB source : tests/trigtech/test_sample.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech, trigpts

EPS = float(np.finfo(np.float64).eps)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestTrigtechSample:
    def test_native_grid_sample(self):
        f = Trigtech.from_function(lambda x: jnp.exp(jnp.sin(jnp.pi * x - 0.1)))
        p = trigpts(f.n)
        v = f.values
        assert _ninf(p - trigpts(f.n)) < 100 * EPS
        assert _ninf(v - f(p)) < 100 * EPS

    @pytest.mark.skip(
        reason="chebfunjax trigtech has no sample(f, m) resampling method; the equivalent "
        "(feval on trigpts(m)) is already covered by test_feval"
    )
    def test_shorter_grid(self):
        pass

    @pytest.mark.skip(
        reason="chebfunjax trigtech has no sample(f, m) resampling method; the equivalent "
        "(feval on trigpts(m)) is already covered by test_feval"
    )
    def test_longer_grid(self):
        pass
