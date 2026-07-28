"""Port of MATLAB Chebfun tests/chebfun/test_constructor_equi.m (Fable 5).

The ``equi=True`` flag interprets numeric data as samples on an
equispaced grid ``linspace(a, b, N)`` and builds a Floater-Hormann
rational interpolant (FUNQUI) which is then resolved as a Chebfun.

Provenance
----------
MATLAB source : tests/chebfun/test_constructor_equi.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(6178)
XX = 2 * RNG.random(100) - 1


def _vscale(g):
    return float(g.vscale)


class TestChebfunConstructorEqui:
    def test_arrayvalued_constant_columns(self):
        # pass(1): a 1-by-10 row of samples -> [Inf, 10] with each column
        # the constant sample value.
        v = np.cos(np.linspace(-1, 1, 10))
        g = cj.chebfun(v.reshape(1, -1), equi=True)
        assert g.size() == (math.inf, 10)
        got = np.asarray(g(jnp.asarray(XX)))
        assert np.linalg.norm(got - np.tile(v, (100, 1))) < 10 * _vscale(g) * EPS

    def test_short_columns(self):
        # pass(2) and pass(3): 3- and 2-sample single columns -> [Inf, 1].
        v = np.cos(np.linspace(-1, 1, 10))
        g = cj.chebfun(v[:3].reshape(-1, 1), equi=True)
        assert g.size() == (math.inf, 1)
        g = cj.chebfun(v[:2].reshape(-1, 1), equi=True)
        assert g.size() == (math.inf, 1)

    def test_matrices_tall_and_wide(self):
        # pass(4) and pass(5): (10, 10) and (10, 11) -> [Inf, 10]/[Inf, 11].
        v = np.cos(np.linspace(-1, 1, 10))
        g = cj.chebfun(np.tile(v.reshape(-1, 1), (1, 10)), equi=True)
        assert g.size() == (math.inf, 10)
        g = cj.chebfun(np.tile(v.reshape(-1, 1), (1, 11)), equi=True)
        assert g.size() == (math.inf, 11)

    @pytest.mark.parametrize("data, exact", [
        ([-1.0, 0.0, 1.0], lambda t: t),            # pass(6)
        ([-3.0, -1.0, 1.0, 3.0], lambda t: 3 * t),  # pass(7)
        ([-1e5, 1e5], lambda t: 1e5 * t),           # pass(8)
        ([0.0, 1.0, 0.0], lambda t: 1 - t ** 2),    # pass(9)
    ])
    def test_lines_and_parabola(self, data, exact):
        g = cj.chebfun(np.array(data).reshape(-1, 1), equi=True)
        got = np.asarray(g(jnp.asarray(XX))).reshape(-1)
        assert np.linalg.norm(got - exact(XX)) < 10 * _vscale(g) * EPS

    def test_scalar_constant(self):
        # pass(10): a single scalar -> constant Chebfun.
        u = float(XX[-1])
        g = cj.chebfun(u, equi=True)
        got = np.asarray(g(jnp.asarray(XX))).reshape(-1)
        assert np.linalg.norm(got - u) < 10 * _vscale(g) * EPS

    def test_equi_with_function_handle_errors(self):
        # pass(11): 'equi' with a function handle (adaptive) is an error.
        with pytest.raises(ValueError):
            cj.chebfun(lambda x: jnp.exp(x), equi=True)
