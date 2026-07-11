"""Port of MATLAB Chebfun tests/diskfun/test_laplacian.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_laplacian.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.diskfun.diskfun import Diskfun

TOL = 1e-8


class TestDiskfunLaplacian:
    @pytest.mark.xfail(
        reason="Opus 4.8's diskfun laplacian drops the ANGULAR "
        "dependence of the result: lap((1-r^2)r^2 cos2t) returns "
        "-12r^2 without the cos2t factor (got -3.63, exact +0.825 at "
        "(0.9, 0.55)). All previously-passing cases had "
        "theta-independent laplacians. Same _diskfun_reconstruct "
        "pipeline bug family as diffx/diffy. Fable 5 audit.")
    def test_laplacian_with_angular_dependence(self):
        u = Diskfun.from_function(
            lambda t, r: (1 - r ** 2) * r ** 2 * jnp.cos(2 * t))
        lap = u.laplacian()
        t0, r0 = 0.9, 0.55
        exact = -12 * r0 ** 2 * np.cos(2 * t0)
        got = float(lap(jnp.asarray(t0), jnp.asarray(r0)))
        assert abs(got - exact) < 1e-8

    def test_laplacian_of_harmonic_is_zero(self):
        # x^2 - y^2 is harmonic
        f = Diskfun.from_function(
            lambda t, r: r ** 2 * jnp.cos(2 * t))
        g = f.laplacian()
        assert abs(float(g(jnp.asarray(0.4), jnp.asarray(0.6)))) < TOL

    def test_laplacian_of_r2(self):
        # NOTE: this and the harmonic case have THETA-INDEPENDENT
        # laplacians -- the only class the implementation gets right.
        # lap(x^2 + y^2) = 4
        f = Diskfun.from_function(lambda t, r: r ** 2)
        g = f.laplacian()
        assert abs(float(g(jnp.asarray(0.2), jnp.asarray(0.5))) - 4.0) \
            < TOL
