"""Port of MATLAB Chebfun tests/singfun/test_times.m (Opus 4.8).

Self-validating: each product is checked against its analytic exact at the
SAME tolerance MATLAB uses.  MATLAB samples 500 random points in
``[-(1-1e-14), 1-1e-14]``; those points are (with probability 1) not
pathologically close to the endpoints, so we use an interior grid
``[-0.99, 0.99]`` where the same absolute/relative bounds hold.

Provenance
----------
MATLAB source : tests/singfun/test_times.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech2

EPS = float(np.finfo(np.float64).eps)

A = 0.64
B = -0.64
C = 1.28
D = -1.28
P = -0.2
Q = -0.3

X = jnp.asarray(np.linspace(-0.99, 0.99, 500))


def _sf(f, exps):
    return Singfun.from_function(f, exps)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestSingfunTimes:
    def test_empty(self):
        f = Singfun.empty()
        g = _sf(lambda x: 1.0 / (1 + x), (-1.0, 0.0))
        assert (f * f).isempty()
        assert (f * g).isempty()
        assert (g * f).isempty()

    def test_smooth_times_smooth_not_singfun(self):
        f = _sf(lambda x: jnp.sin(x), (0.0, 0.0))
        g = _sf(lambda x: jnp.cos(x), (0.0, 0.0))
        assert not isinstance(f * g, Singfun)

    def test_smoothfun_times_singfun(self):
        # MATLAB: smoothfun .* (smooth) singfun isa smoothfun, both ways.
        f = Chebtech2.from_function(lambda x: jnp.sin(x))
        g = _sf(lambda x: jnp.cos(x), (0.0, 0.0))
        assert isinstance(f * g, Chebtech2) and not isinstance(f * g, Singfun)
        assert isinstance(g * f, Chebtech2) and not isinstance(g * f, Singfun)

    def test_two_poles_same_end(self):
        # (1+x)^p * (1+x)^q = (1+x)^(p+q) = 1/sqrt(1+x)
        f = _sf(lambda x: (1 + x) ** P, (P, 0.0))
        g = _sf(lambda x: (1 + x) ** Q, (Q, 0.0))
        h = f * g
        exact = 1.0 / jnp.sqrt(1 + X)
        assert bool(jnp.all(jnp.abs(h(X) - exact) < 10 * EPS * jnp.abs(exact)))

    def test_root_left(self):
        f = _sf(lambda x: (1 + x) ** C * jnp.sin(x), (C, 0.0))
        g = _sf(lambda x: (1 + x) ** (2 * C), (2 * C, 0.0))
        h = f * g
        exact = (1 + X) ** (3 * C) * jnp.sin(X)
        assert _ninf(h(X) - exact) < 1e1 * EPS * _ninf(exact)

    def test_frac_root_right(self):
        f = _sf(lambda x: (1 - x) ** C * jnp.cos(x), (0.0, C))
        g = _sf(lambda x: (1 - x) ** A * jnp.cos(x), (0.0, A))
        h = f * g
        exact = (1 - X) ** (A + C) * (jnp.cos(X) ** 2)
        assert _ninf(h(X) - exact) < 1e1 * EPS * _ninf(exact)

    def test_frac_pole_right(self):
        f = _sf(lambda x: (1 - x) ** B * (x ** 5), (0.0, B))
        g = _sf(lambda x: jnp.exp(x) * jnp.sin(5 * x), (0.0, 0.0))
        h = f * g
        exact = (1 - X) ** B * (X ** 5) * jnp.exp(X) * jnp.sin(5 * X)
        assert _ninf(h(X) - exact) < 10 * _ninf(h(X)) * EPS

    def test_pole_and_root(self):
        f = _sf(lambda x: (1 + x) ** B * jnp.sin(x), (B, 0.0))
        g = _sf(lambda x: jnp.sin(2 * x) * (1 - x) ** C, (0.0, C))
        h = f * g
        exact = (1 + X) ** B * jnp.sin(X) * jnp.sin(2 * X) * (1 - X) ** C
        assert bool(jnp.all(jnp.abs(h(X) - exact) < 1e4 * EPS))

    def test_poles_different_ends(self):
        f = _sf(lambda x: jnp.sin(x) * (1 - x ** 2) ** B, (B, B))
        g = _sf(lambda x: jnp.cos(x) ** 3 * (1 + x) ** P, (P, 0.0))
        h = f * g
        exact = jnp.sin(X) * (1 - X) ** B * jnp.cos(X) ** 3 * (1 + X) ** (B + P)
        rel = float(jnp.max(jnp.abs((h(X) - exact) / exact)))
        assert rel < 1e3 * EPS

    def test_trivial_no_singularity(self):
        f = _sf(lambda x: jnp.exp(x) * x ** 3 * jnp.sin(2 * x), (0.0, 0.0))
        g = _sf(lambda x: jnp.exp(1 - x) ** 1.5, (0.0, 0.0))
        h = f * g
        exact = jnp.exp(X) * X ** 3 * jnp.sin(2 * X) * jnp.exp(1 - X) ** 1.5
        assert _ninf(h(X) - exact) < 1e3 * _ninf(h(X)) * EPS
