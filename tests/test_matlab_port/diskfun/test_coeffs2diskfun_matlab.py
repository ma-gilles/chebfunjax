"""Port of MATLAB Chebfun tests/diskfun/test_coeffs2diskfun.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_coeffs2diskfun.m
Chebfun commit: 7574c77

Cartesian ``@(x,y)`` handles from MATLAB are written directly in polar
coordinates ``(theta, r)`` with ``x = r cos(theta)``, ``y = r sin(theta)``.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun

_EPS = float(jnp.finfo(jnp.float64).eps)
_TOL = 10 * _EPS


def _df(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Diskfun.from_function(fn)


class TestDiskfunCoeffs2diskfun:
    def test_zero_scalar(self):
        # pass(1): f = coeffs2diskfun(0); iszero(f)
        f = Diskfun.coeffs2diskfun(0)
        assert f.iszero()

    def test_roundtrip_r_squared(self):
        # pass(2): f = r^2 (polar); g = coeffs2diskfun(coeffs2(f))
        f = _df(lambda t, r: r**2)
        g = Diskfun.coeffs2diskfun(f.coeffs2())
        assert float((f - g).norm()) < _TOL

    def test_roundtrip_r_sin_theta(self):
        # pass(3): f = r sin(theta) (polar)
        f = _df(lambda t, r: r * jnp.sin(t))
        g = Diskfun.coeffs2diskfun(f.coeffs2())
        assert float((f - g).norm()) < _TOL

    def test_roundtrip_gaussian(self):
        # pass(4): f = exp(-10*((x-a)^2+(y-a)^2)), a = 0.5/sqrt(2)
        a = 0.5 / np.sqrt(2.0)
        f = _df(
            lambda t, r: jnp.exp(
                -10 * ((r * jnp.cos(t) - a) ** 2 + (r * jnp.sin(t) - a) ** 2)
            )
        )
        g = Diskfun.coeffs2diskfun(f.coeffs2())
        # MATLAB pass(4) just evaluates norm(f-g) (no explicit bound); we
        # keep it meaningful and require the round trip to be tight.
        assert float((f - g).norm()) < _TOL

    def test_explicit_matrix_r_sin_theta(self):
        # pass(5): c = 1i/2 * [0 0 0; 1 0 -1]  ->  r sin(theta)
        c = 1j / 2.0 * np.array([[0, 0, 0], [1, 0, -1]], dtype=np.complex128)
        f = _df(lambda t, r: r * jnp.sin(t))
        assert float((f - Diskfun.coeffs2diskfun(c)).norm()) < _TOL

    def test_explicit_matrix_cubic(self):
        # pass(6): r^3 cos(3t) + r^2 sin(t)^2
        c = (1.0 / 8.0) * np.array(
            [
                [0, -1, 0, 2, 0, -1, 0],
                [3, 0, 0, 0, 0, 0, 3],
                [0, -1, 0, 2, 0, -1, 0],
                [1, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.complex128,
        )
        f = _df(lambda t, r: r**3 * jnp.cos(3 * t) + r**2 * jnp.sin(t) ** 2)
        assert float((f - Diskfun.coeffs2diskfun(c)).norm()) < _TOL
