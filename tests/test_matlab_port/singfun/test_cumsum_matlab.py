"""Port of MATLAB Chebfun tests/singfun/test_cumsum.m (Opus 4.8).

All six original predicates with original bounds. The seed6178 MT19937
primitive stream is checked against the retained native MATLAB input capture.
The source D=2 arithmetic is then applied in its original order.

Provenance
----------
MATLAB source : tests/singfun/test_cumsum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np  # uses-numpy: native-verified source RNG test inputs

from chebfunjax.fun.singfun import Singfun

EPS = float(np.finfo(np.float64).eps)

A = 0.64
B = -0.64
C = 1.28
D = -1.28

_REF = json.loads((Path(__file__).parents[1] / "chebfun/fixtures/logical_source_matlab.json").read_text())
_R = np.random.RandomState(6178).rand(100)
assert np.array_equal(2 * _R - 1, _REF["x"])
X = jnp.asarray(2 * (1 - 10 ** (-2)) * _R - (1 - 10 ** (-2)))


def _sf(f, exps):
    return Singfun.from_function(f, exps)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestSingfunCumsum:
    def test_frac_pole_left(self):
        # fractional pole (order > -1) at the left endpoint
        f = _sf(lambda x: (1 + x) ** B, (B, 0.0))
        g = f.cumsum()
        exact = (1 + X) ** (B + 1) / (B + 1)
        assert _ninf(g(X) - exact) < 1e1 * EPS * _ninf(exact)

    def test_frac_pole_right_order_lt_m1(self):
        # fractional pole with order < -1 at the right endpoint
        f = _sf(lambda x: (1 - x) ** D, (0.0, D))
        g = f.cumsum()
        exact = -(1 - X) ** (D + 1) / (D + 1) + 2 ** (D + 1) / (D + 1)
        assert _ninf(g(X) - exact) < 1e3 * EPS * _ninf(exact)

    def test_frac_root_left(self):
        f = _sf(lambda x: (1 + x) ** A, (A, 0.0))
        g = f.cumsum()
        exact = (1 + X) ** (A + 1) / (A + 1)
        assert _ninf(g(X) - exact) < EPS * _ninf(exact)

    def test_integer_pole_right(self):
        f = _sf(lambda x: (1 - x) ** (-4.0), (0.0, -4.0))
        g = f.cumsum()
        exact = (1 - X) ** (-3.0) / 3 - 2 ** (-3.0) / 3
        assert _ninf(g(X) - exact) < 1e2 * EPS * _ninf(exact)

    def test_no_closed_form_pole_left(self):
        f = _sf(lambda x: jnp.cos(x ** 2 + 3) * ((1 + x) ** B), (B, 0.0))
        u = f.cumsum()
        dom = [-1 + 10 ** (-2), 1]
        scl = (dom[1] - dom[0]) / 2
        g = u.restrict(dom)
        v = f.restrict(dom)
        h = scl * v.cumsum()
        h = h - h(1) + u(1)
        exact = h(X)
        assert _ninf(g(X) - exact) < 1e4 * EPS * _ninf(exact)

    def test_no_closed_form_root_left(self):
        f = _sf(lambda x: jnp.cos(jnp.sin(x)) * (1 + x) ** C, (C, 0.0))
        u = f.cumsum()
        dom = [-1 + 10 ** (-2), 1]
        scl = (dom[1] - dom[0]) / 2
        g = u.restrict(dom)
        v = f.restrict(dom)
        h = scl * v.cumsum()
        h = h - h(1) + u(1)
        exact = h(X)
        assert _ninf(g(X) - exact) < 1e2 * EPS * _ninf(exact)
