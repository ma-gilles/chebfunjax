"""Port of MATLAB Chebfun tests/chebfun2/test_composition_operators.m (Fable 5).

FIXED (Fable 5, chebfun2/3 skip sweep): Chebfun2 has the elementwise
composition operators (cos/cosh/sin/sinh/...), ``compose`` with a
Chebfun, an array-valued Chebfun (giving a Chebfun2v), and a Chebfun2,
plus ``isPeriodicTech``.

MATLAB pass(11-13, 15, 16) build 'trig' Chebfun2s; the trigonometric
tech option is not wired into the Chebfun2 constructor (the
SeparableApprox techs field exists but only 'cheb' is reachable), so
those five assertions are not ported -- see test_trig_matlab.py.

Provenance
----------
MATLAB source : tests/chebfun2/test_composition_operators.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

EPS = float(np.finfo(np.float64).eps)
TOL = 1000 * EPS


def _base():
    return Chebfun2.from_function(
        lambda x, y: jnp.cos(x * y) + jnp.sin(x * y) + y - 0.1)


def _base_np(x, y):
    return np.cos(x * y) + np.sin(x * y) + y - 0.1


class TestChebfun2Compositionoperators:
    def test_multiplication(self):
        # pass(1): f .* sin((x-.1)(y+.4)).
        f = _base()
        x = Chebfun2.from_function(lambda x, y: x)
        y = Chebfun2.from_function(lambda x, y: y)
        g = Chebfun2.from_function(
            lambda x, y: (jnp.cos(x * y) + jnp.sin(x * y) + y - 0.1)
            * jnp.sin((x - 0.1) * (y + 0.4)))
        h = f * ((x - 0.1) * (y + 0.4)).sin()
        assert float((g - h).norm()) < TOL

    def test_cos(self):
        # pass(2)
        g = Chebfun2.from_function(
            lambda x, y: jnp.cos(jnp.cos(x * y) + jnp.sin(x * y) + y - 0.1))
        assert float((g - _base().cos()).norm()) < TOL

    def test_cosh(self):
        # pass(3)
        g = Chebfun2.from_function(
            lambda x, y: jnp.cosh(jnp.cos(x * y) + jnp.sin(x * y) + y - 0.1))
        assert float((g - _base().cosh()).norm()) < TOL

    def test_sin(self):
        # pass(4)
        g = Chebfun2.from_function(
            lambda x, y: jnp.sin(jnp.cos(x * y) + jnp.sin(x * y) + y - 0.1))
        assert float((g - _base().sin()).norm()) < TOL

    def test_sinh(self):
        # pass(5)
        g = Chebfun2.from_function(
            lambda x, y: jnp.sinh(jnp.cos(x * y) + jnp.sin(x * y) + y - 0.1))
        assert float((g - _base().sinh()).norm()) < TOL

    def test_multiple_operations(self):
        # pass(6, 7): f+f+f == 3f and f.*f == f.^2 on [-1 2 -1 1].
        f = Chebfun2.from_function(lambda x, y: jnp.sin(10 * x * y),
                                   domain=(-1.0, 2.0, -1.0, 1.0))
        assert float((f + f + f - 3 * f).norm()) < 100 * TOL
        assert float((f * f - f ** 2).norm()) < TOL

    def test_compose_with_chebfun(self):
        # pass(8, 9): compose(x + y, t -> t^2) on a Chebfun defined over
        # the range of f; the result is not periodic.
        f = Chebfun2.from_function(lambda x, y: x + y)
        g = cj.chebfun(lambda t: t ** 2, domain=(-2.0, 2.0))
        h = f.compose(g)
        h_true = Chebfun2.from_function(lambda x, y: (x + y) ** 2)
        assert float((h - h_true).norm()) < TOL
        assert not h.isPeriodicTech()

    def test_compose_with_array_valued_chebfun(self):
        # pass(10): a two-column Chebfun gives a Chebfun2v.
        f = Chebfun2.from_function(lambda x, y: x + y)
        G = cj.chebfun(lambda t: jnp.stack([t, t ** 2], axis=-1),
                       domain=(-2.0, 2.0))
        H = f.compose(G)
        H_true = Chebfun2v.from_functions(lambda x, y: x + y,
                                          lambda x, y: (x + y) ** 2)
        assert H.n_components == 2
        assert float((H - H_true).norm()) < TOL

    def test_compose_complex_with_chebfun2(self):
        # pass(14): compose(x + 1i*y, g) evaluates g at (x, y).
        f = Chebfun2.from_function(lambda x, y: x + 1j * y)
        g = Chebfun2.from_function(lambda x, y: x ** 2 + y ** 2)
        h_true = Chebfun2.from_function(lambda x, y: x ** 2 + y ** 2)
        assert float((f.compose(g) - h_true).norm()) < TOL
