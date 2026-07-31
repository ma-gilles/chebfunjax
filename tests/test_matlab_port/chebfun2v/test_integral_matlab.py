"""Port of MATLAB Chebfun tests/chebfun2v/test_integral.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_integral.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

TOL = 100 * float(np.finfo(np.float64).eps)


class TestChebfun2vIntegral:
    def test_circulation_unit_circle(self):
        # pass(1): integral of (-y, x) along exp(1i t), t in [-pi, pi] = 2*pi.
        F = Chebfun2v.from_functions(lambda x, y: -y + 0 * x,
                                     lambda x, y: x + 0 * y)
        c = cj.chebfun(lambda t: jnp.exp(1j * t),
                       domain=[-np.pi, np.pi])
        I = float(np.real(np.asarray(F.integral(c))))
        assert abs(I - 2 * np.pi) < TOL

    def test_quarter_circle(self):
        # pass(2): integral of (xy, y) along exp(1i t), t in [0, pi/2] = 1/6.
        F = Chebfun2v.from_functions(lambda x, y: x * y,
                                     lambda x, y: y + 0 * x)
        c = cj.chebfun(lambda t: jnp.exp(1j * t),
                       domain=[0.0, np.pi / 2])
        I = float(np.real(np.asarray(F.integral(c))))
        assert abs(I - 1.0 / 6.0) < TOL
