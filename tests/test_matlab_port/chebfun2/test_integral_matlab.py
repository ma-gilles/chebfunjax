"""Port of MATLAB Chebfun tests/chebfun2/test_integral.m (Fable 5).

FIXED (Fable 5, chebfun2/3 skip sweep): ``Chebfun2.integral(curve)``
now exists, computing ``sum(f(c(t)) * |c'(t)|)``.

Provenance
----------
MATLAB source : tests/chebfun2/test_integral.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.special import jv

import chebfunjax as cj
from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS


def _circle():
    return cj.chebfun(lambda t: jnp.exp(1j * t), domain=(0.0, 2 * np.pi))


class TestChebfun2Integral:
    def test_arclength_of_unit_circle(self):
        # pass(1): integrating 1 around the unit circle gives its length.
        f = Chebfun2.from_function(lambda x, y: 1.0 + 0 * x)
        assert abs(float(f.integral(_circle())) - 2 * np.pi) < TOL

    def test_cos_x_around_circle(self):
        # pass(2): int_0^{2pi} cos(cos t) dt = 2 pi J_0(1).
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x))
        exact = 2 * np.pi * jv(0, 1)
        assert abs(float(f.integral(_circle())) - exact) < TOL

    def test_cos_xy_around_circle(self):
        # pass(3): compared against the equivalent 1D quadrature.
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        exact = float(cj.chebfun(
            lambda t: jnp.cos(jnp.cos(t) * jnp.sin(t)),
            domain=(0.0, 2 * np.pi)).sum())
        assert abs(float(f.integral(_circle())) - exact) < TOL

    def test_integral_along_real_curve(self):
        # pass(4): a real-valued curve c(t) = t traces the x-axis, so
        # only the cos(x) part contributes.
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x) + jnp.sin(y))
        c = cj.chebfun(lambda t: t)
        exact = float(cj.chebfun(jnp.cos).sum())
        assert abs(float(f.integral(c)) - exact) < TOL

    def test_integral_without_curve_is_integral2(self):
        # MATLAB integral(f) with one argument is another way to write
        # integral2(f).
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        assert abs(float(f.integral()) - float(f.integral2())) < TOL
