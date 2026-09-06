"""Port of MATLAB Chebfun tests/domain/test_polyfit.m (Fable 5).

Provenance
----------
MATLAB source : tests/domain/test_polyfit.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import polyfit
from chebfunjax.utils.quadrature import chebpts

jax.config.update("jax_enable_x64", True)

EPS = np.finfo(float).eps


def _vs(f):
    return float(np.max(np.atleast_1d(np.asarray(f.vscale))))


def _err(f, x, y):
    return np.linalg.norm(np.asarray(f(jnp.asarray(x))) - y)


class TestDomainPolyfit:
    def test_all_matlab_assertions(self):
        rng = np.random.RandomState(7681)
        dom = (-1, 1)
        x = np.linspace(-1, 1, 1000)
        y = x ** 2
        f = polyfit(x, y, 2, dom)
        assert _err(f, x, y) < 100 * EPS * _vs(f)                             # pass(1)
        x = np.linspace(-1, 1, 100)
        y = x ** 3
        f = polyfit(x, y, 3, dom)
        assert _err(f, x, y) < 5e5 * EPS * _vs(f)                             # pass(2)
        x = np.linspace(-1, 1, 50)
        y = 2 * x + 4 * x ** 2 + x ** 4
        f = polyfit(x, y, 4, dom)
        assert _err(f, x, y) < 5e5 * EPS * _vs(f)                             # pass(3)
        x = np.asarray(chebpts(1000))
        y = 2 * x + 4 * x ** 2 + x ** 4
        f = polyfit(x, y, 4, dom)
        assert _err(f, x, y) < 5e5 * EPS * _vs(f)                             # pass(4)
        x = np.linspace(-1, 1, 10)
        y = -5 + 10 * rng.rand(10)
        f = polyfit(x, y, 12, dom)
        assert _err(f, x, y) < 10 * EPS * _vs(f)                              # pass(5)
        x = np.linspace(-1, 1, 1000)
        y = x ** 2
        f = polyfit(x, y, 3, dom)
        assert _err(f, x, y) < 100 * EPS * _vs(f)                             # pass(6)
        x = np.linspace(0, 1000, 1000)
        y = x ** 2
        f = polyfit(x, y, 2, (0, 1000))
        assert _err(f, x, y) < 100 * EPS * _vs(f)                             # pass(7)
        x = np.linspace(-1, 1, 10)
        y = np.column_stack([x ** 2, x ** 3])
        f = polyfit(x, y, 3, dom)
        err = np.linalg.norm(np.asarray(f(jnp.asarray(x))) - y, axis=0)
        assert np.all(err < 10 * _vs(f) * EPS)                                # pass(8)
