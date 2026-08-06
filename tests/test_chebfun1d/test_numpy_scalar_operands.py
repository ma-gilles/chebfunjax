"""A numpy scalar on the left of a Chebfun.

``Chebfun`` defined neither ``__array_ufunc__`` nor ``__array_priority__``,
so numpy tried to broadcast the Chebfun instead of returning
NotImplemented and letting Python call the reflected operator. A plain
Python float worked, a numpy scalar raised:

    2.0 - f              ok
    np.float64(2) - f    ValueError: setting an array element with a sequence
    np.sin(0.5) - f      ValueError  (np.sin RETURNS np.float64)

The third form is the common one: any expression transcribed from MATLAB
like ``2*sin(psi) - b*u(1)`` computes its coefficient with numpy and so
hits this. Found while porting ode-nonlin/Droplets, whose prescribed-
volume constraint is exactly that shape.
"""
from __future__ import annotations

import numpy as np
import pytest

from chebfunjax import chebfun
from chebfunjax.chebfun1d.chebfun import Chebfun


@pytest.fixture
def f():
    return chebfun(lambda x: x, domain=(-1.0, 1.0))


@pytest.mark.parametrize("lhs", [
    np.float64(2.0),
    np.sin(0.5),          # np.sin returns np.float64 -- the common case
    np.array(2.0),
    np.int64(2),
])
class TestNumpyScalarOnTheLeft:
    def test_sub(self, f, lhs):
        g = lhs - f
        assert isinstance(g, Chebfun)
        x = np.linspace(-1, 1, 16)
        assert np.asarray(g(x)) == pytest.approx(float(lhs) - x, abs=1e-12)

    def test_mul(self, f, lhs):
        g = lhs * f
        assert isinstance(g, Chebfun)
        x = np.linspace(-1, 1, 16)
        assert np.asarray(g(x)) == pytest.approx(float(lhs) * x, abs=1e-12)

    def test_add(self, f, lhs):
        g = lhs + f
        assert isinstance(g, Chebfun)
        x = np.linspace(-1, 1, 16)
        assert np.asarray(g(x)) == pytest.approx(float(lhs) + x, abs=1e-12)


def test_div_by_a_chebfun(f):
    g = np.float64(2.0) / (f + 2.0)
    assert isinstance(g, Chebfun)
    x = np.linspace(-1, 1, 16)
    assert np.asarray(g(x)) == pytest.approx(2.0 / (x + 2.0), abs=1e-12)


def test_chebfun_on_the_left_still_works(f):
    g = f * np.float64(2.0)
    assert isinstance(g, Chebfun)
    x = np.linspace(-1, 1, 16)
    assert np.asarray(g(x)) == pytest.approx(2.0 * x, abs=1e-12)


def test_evaluation_still_returns_an_array(f):
    # __array_ufunc__ = None must not stop np.asarray on the VALUES.
    v = np.asarray(f(np.linspace(-1, 1, 5)))
    assert isinstance(v, np.ndarray)
    assert v == pytest.approx(np.linspace(-1, 1, 5), abs=1e-12)


def test_the_droplets_shaped_expression(f):
    # pi*b*(2*sin(psib) - b*u(1)) - v0, the constraint that found this.
    psib, v0 = -np.pi, 10.0
    b = chebfun(lambda x: 0.0 * x + 1.5, domain=(-1.0, 1.0))
    expr = np.pi * b * (2 * np.sin(psib) - b * float(f(np.float64(1.0)))) - v0
    assert isinstance(expr, Chebfun)
    want = np.pi * 1.5 * (2 * np.sin(psib) - 1.5 * 1.0) - v0
    assert float(expr(np.float64(0.0))) == pytest.approx(want, abs=1e-10)
