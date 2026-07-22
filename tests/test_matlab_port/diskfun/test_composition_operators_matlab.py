"""Port of MATLAB Chebfun tests/diskfun/test_composition_operators.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_composition_operators.m
Chebfun commit: 7574c77

Cartesian ``@(x,y)`` handles from MATLAB are written directly in polar
coordinates ``(theta, r)`` with ``x = r cos(theta)``, ``y = r sin(theta)``.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.diskfun.diskfun import Diskfun
from chebfunjax.diskfun.diskfunv import Diskfunv
from chebfunjax.domain import Domain

_EPS = float(jnp.finfo(jnp.float64).eps)
_TOL = 1e3 * _EPS


def _df(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Diskfun.from_function(fn)


def _x(t, r):
    return r * jnp.cos(t)


def _y(t, r):
    return r * jnp.sin(t)


class TestDiskfunCompositionOperators:
    def test_multiplication(self):
        f = _df(lambda t, r: jnp.cos(_x(t, r) * _y(t, r)) + jnp.sin(_x(t, r) * _y(t, r)))
        g = _df(
            lambda t, r: (jnp.cos(_x(t, r) * _y(t, r)) + jnp.sin(_x(t, r) * _y(t, r)))
            * jnp.sin((_x(t, r) - 0.1) * (_y(t, r) + 0.4))
        )
        prod = f * _df(lambda t, r: jnp.sin((_x(t, r) - 0.1) * (_y(t, r) + 0.4)))
        assert float((g - prod).norm()) < _TOL

    def test_cos(self):
        f = _df(lambda t, r: jnp.cos(_x(t, r) * _y(t, r)) + jnp.sin(_x(t, r) * _y(t, r)))
        g = _df(lambda t, r: jnp.cos(jnp.cos(_x(t, r) * _y(t, r)) + jnp.sin(_x(t, r) * _y(t, r))))
        assert float((g - f.cos()).norm()) < _TOL

    def test_cosh(self):
        f = _df(lambda t, r: jnp.cos(_x(t, r) * _y(t, r)) + jnp.sin(_x(t, r) * _y(t, r)))
        g = _df(lambda t, r: jnp.cosh(jnp.cos(_x(t, r) * _y(t, r)) + jnp.sin(_x(t, r) * _y(t, r))))
        assert float((g - f.cosh()).norm()) < _TOL

    def test_sin(self):
        f = _df(lambda t, r: jnp.cos(_x(t, r) * _y(t, r)) + jnp.sin(_x(t, r) * _y(t, r)))
        g = _df(lambda t, r: jnp.sin(jnp.cos(_x(t, r) * _y(t, r)) + jnp.sin(_x(t, r) * _y(t, r))))
        assert float((g - f.sin()).norm()) < _TOL

    def test_sinh(self):
        f = _df(lambda t, r: jnp.cos(_x(t, r) * _y(t, r)) + jnp.sin(_x(t, r) * _y(t, r)))
        g = _df(lambda t, r: jnp.sinh(jnp.cos(_x(t, r) * _y(t, r)) + jnp.sin(_x(t, r) * _y(t, r))))
        assert float((g - f.sinh()).norm()) < _TOL

    def test_multiple_operations(self):
        f = _df(lambda t, r: jnp.sin(jnp.pi * _x(t, r) * _y(t, r)))
        assert float((f + f + f - 3 * f).norm()) < 100 * _TOL
        assert float((f * f - f**2).norm()) < _TOL

    def test_compose_chebfun_one_column(self):
        # pass(8): f = x + y; g = chebfun(t^2, [-2,2]); h = compose(f, g)
        f = _df(lambda t, r: _x(t, r) + _y(t, r))
        g = Chebfun.from_function(lambda t: t**2, domain=Domain((-2.0, 2.0)))
        h = f.compose(g)
        h_true = _df(lambda t, r: (_x(t, r) + _y(t, r)) ** 2)
        assert float((h - h_true).norm()) < _TOL

    def test_compose_chebfun_two_columns(self):
        # pass(9): G = chebfun([t^2, exp(t)], [-2,2]); H = compose(f, G) -> diskfunv
        f = _df(lambda t, r: _x(t, r) + _y(t, r))
        G = Quasimatrix.from_functions(
            [lambda t: t**2, lambda t: jnp.exp(t)], domain=Domain((-2.0, 2.0))
        )
        H = f.compose(G)
        assert isinstance(H, Diskfunv)
        H_true = Diskfunv.from_functions(
            lambda t, r: (_x(t, r) + _y(t, r)) ** 2,
            lambda t, r: jnp.exp(_x(t, r) + _y(t, r)),
        )
        for hc, tc in zip(H.components, H_true.components):
            assert float((hc - tc).norm()) < _TOL

    def test_compose_chebfun2(self):
        # pass(10): g = chebfun2(x^2 + y^2, [-2,2,-2,2]); h = compose(f, g)
        f = _df(lambda t, r: _x(t, r) + _y(t, r))
        g = Chebfun2.from_function(lambda x, y: x**2 + y**2, domain=(-2.0, 2.0, -2.0, 2.0))
        h = f.compose(g)
        h_true = _df(lambda t, r: (_x(t, r) + _y(t, r)) ** 2)
        assert float((h - h_true).norm()) < _TOL

    def test_compose_chebfun2v(self):
        # pass(11): H = compose(f, [g; g]) -> diskfunv [h_true; h_true]
        f = _df(lambda t, r: _x(t, r) + _y(t, r))
        gv = Chebfun2v.from_functions(
            lambda x, y: x**2 + y**2,
            lambda x, y: x**2 + y**2,
            domain=(-2.0, 2.0, -2.0, 2.0),
        )
        H = f.compose(gv)
        assert isinstance(H, Diskfunv)
        h_true = _df(lambda t, r: (_x(t, r) + _y(t, r)) ** 2)
        for hc in H.components:
            assert float((hc - h_true).norm()) < _TOL
