"""Core coverage for Diskfun structural plus/scale (no re-approximation)."""
from __future__ import annotations

import time

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun


class TestStructuralArithmetic:
    def test_sum_and_scale_evaluate_exactly(self):
        a = Diskfun.harmonic(0, 1)
        b = Diskfun.harmonic(2, 2)
        s = a + b * 2.0
        th = jnp.asarray([0.3, 1.1, -2.0])
        r = jnp.asarray([0.4, 0.8, 0.1])
        want = np.asarray(a(th, r)) + 2 * np.asarray(b(th, r))
        assert float(np.max(np.abs(np.asarray(s(th, r)) - want))) < 1e-13

    def test_sub_and_rsub(self):
        a = Diskfun.harmonic(1, 1)
        d = a - a
        th = jnp.asarray([0.5])
        r = jnp.asarray([0.6])
        assert abs(float(np.asarray(d(th, r))[0])) < 1e-13

    def test_lap_plus_scaled_is_fast_and_finite(self):
        # Regression for the 300s + NaN adaptive re-approximation path.
        u = Diskfun.harmonic(4, 2)
        t0 = time.time()
        resid = u.lap() + u * (11.6 ** 2)
        assert time.time() - t0 < 60
        th = jnp.asarray(np.linspace(-3, 3, 20))
        r = jnp.asarray(np.linspace(0.0, 1.0, 20))
        assert np.all(np.isfinite(np.asarray(resid(th, r))))
