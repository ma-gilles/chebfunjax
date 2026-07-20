"""Port of MATLAB Chebfun tests/diskfun/test_coeffs2.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_coeffs2.m
Chebfun commit: 7574c77

Note: MATLAB assertion 5 (``diskfun(zeros(5,4))``) requires the
values/coefficient-matrix constructor, which chebfunjax does not expose
(see test_coeffs2diskfun_matlab.py, "no coefficient constructor"), so it
is omitted here.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.diskfun.diskfun import Diskfun

_EPS = float(jnp.finfo(jnp.float64).eps)
_TOL = 1000 * _EPS


class TestDiskfunCoeffs2:
    def test_r_squared(self):
        f = Diskfun.from_function(lambda t, r: r**2)
        c = np.array([[0.5], [0.0], [0.5]], dtype=np.complex128)
        got = np.asarray(f.coeffs2())
        assert got.shape == c.shape
        assert np.linalg.norm(got - c) < _TOL

    def test_r_sin_theta(self):
        f = Diskfun.from_function(lambda t, r: r * jnp.sin(t))
        c = 1j / 2.0 * np.array([[0, 0, 0], [1, 0, -1]], dtype=np.complex128)
        got = np.asarray(f.coeffs2())
        assert got.shape == c.shape
        assert np.linalg.norm(got - c) < _TOL

    def test_mixed_low_order(self):
        f = Diskfun.from_function(
            lambda t, r: r**2 + r * (4 * jnp.cos(t) + 2 * jnp.sin(t))
        )
        c = 0.5 * np.array(
            [[0, 1, 0], [4 + 2j, 0, 4 - 2j], [0, 1, 0]], dtype=np.complex128
        )
        got = np.asarray(f.coeffs2())
        assert got.shape == c.shape
        assert np.linalg.norm(got - c) < _TOL

    def test_cubic_angular(self):
        f = Diskfun.from_function(
            lambda t, r: r**3 * jnp.cos(3 * t) + r**2 * jnp.sin(t) ** 2
        )
        c = (1.0 / 8.0) * np.array(
            [
                [0, -1, 0, 2, 0, -1, 0],
                [3, 0, 0, 0, 0, 0, 3],
                [0, -1, 0, 2, 0, -1, 0],
                [1, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.complex128,
        )
        got = np.asarray(f.coeffs2())
        assert got.shape == c.shape
        assert np.linalg.norm(got - c) < _TOL

    @pytest.mark.skip(
        reason="MATLAB assertion 5 needs diskfun(zeros(m,n)) matrix "
        "constructor (not exposed in chebfunjax)"
    )
    def test_zeros_matrix(self):
        pass
