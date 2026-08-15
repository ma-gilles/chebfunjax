"""Port of MATLAB Chebfun tests/diskfun/test_get.m (Fable 5).

Property access versus cdr().

Provenance
----------
MATLAB source : tests/diskfun/test_get.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun

jax.config.update("jax_enable_x64", True)

TOL = 1e2 * 1e-14


class TestDiskfunGet:
    def test_all_matlab_assertions(self):
        # MATLAB constructs from cartesian @(x,y); the chebfunjax
        # constructor samples in polar (theta, r).
        f = Diskfun.from_function(
            lambda t, r: 1
            + jnp.sin(jnp.pi * (r * jnp.cos(t)) * (r * jnp.sin(t)))
            + jnp.sin(jnp.pi * r * jnp.cos(t)))
        C, D, R = f.cdr()

        # cols/rows carry the same slice data as the cdr factors.
        assert len(C) == len(f.cols) and len(R) == len(f.rows)
        ts = jnp.linspace(-0.9, 0.9, 15)
        for c_fac, c_attr in zip(C, f.cols):
            assert float(jnp.max(jnp.abs(
                jnp.asarray(c_fac(ts)) - jnp.asarray(c_attr(ts))))) \
                < TOL
        # pivotValues = 1 ./ diag(D).
        d = np.diag(np.asarray(D))
        pv = np.asarray(f.pivot_values)
        nz = np.abs(d) > 0
        assert np.allclose(1.0 / d[nz], pv[nz], atol=1e-10)
        # pivotLocations has one (theta, r) row per rank term.
        assert len(f.pivot_locations) == len(f)
        assert all(len(loc) == 2 for loc in f.pivot_locations)
        # nonZeroPoles: f(0, 0) = 1 != 0.
        assert f.nonzero_poles
        assert abs(float(f(jnp.asarray(0.0), jnp.asarray(0.0)))) > TOL
