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

import json
import math
import struct
from pathlib import Path

import jax.numpy as jnp
import pytest

import chebfunjax as cj

EPS = float(jnp.finfo(jnp.float64).eps)
# Accepted primitive MATLAB query fixture; exact native seed6178 expression.
_fixture = Path(__file__).resolve().parents[2] / 'fixtures/trig_times_rng6178_2025b.json'
_words = json.loads(_fixture.read_text())['query_words']
XX = jnp.asarray([struct.unpack('>d', bytes.fromhex(word))[0] for word in _words])


def _vscale(g):
    return float(g.vscale)


class TestChebfunConstructorEqui:
    def test_arrayvalued_constant_columns(self):
        # pass(1): a 1-by-10 row of samples -> [Inf, 10] with each column
        # the constant sample value.
        v = jnp.cos(jnp.linspace(-1, 1, 10))
        g = cj.chebfun(v.reshape(1, -1), equi=True)
        assert g.size() == (math.inf, 10)
        got = g(XX)
        assert jnp.linalg.norm(got - jnp.tile(v, (100, 1)), ord=2) < 10 * _vscale(g) * EPS

    def test_short_columns(self):
        # pass(2) and pass(3): 3- and 2-sample single columns -> [Inf, 1].
        v = jnp.cos(jnp.linspace(-1, 1, 10))
        g = cj.chebfun(v[:3].reshape(-1, 1), equi=True)
        assert g.size() == (math.inf, 1)
        g = cj.chebfun(v[:2].reshape(-1, 1), equi=True)
        assert g.size() == (math.inf, 1)

    def test_matrices_tall_and_wide(self):
        # pass(4) and pass(5): (10, 10) and (10, 11) -> [Inf, 10]/[Inf, 11].
        v = jnp.cos(jnp.linspace(-1, 1, 10))
        g = cj.chebfun(jnp.tile(v.reshape(-1, 1), (1, 10)), equi=True)
        assert g.size() == (math.inf, 10)
        g = cj.chebfun(jnp.tile(v.reshape(-1, 1), (1, 11)), equi=True)
        assert g.size() == (math.inf, 11)

    @pytest.mark.parametrize("data, expression", [
        ([-1.0, 0.0, 1.0], "x"),            # pass(6)
        ([-3.0, -1.0, 1.0, 3.0], "3*x"),  # pass(7)
        ([-1e5, 1e5], "1e5*x"),           # pass(8)
        ([0.0, 1.0, 0.0], "1 - x.^2"),    # pass(9)
    ])
    def test_lines_and_parabola(self, data, expression):
        g = cj.chebfun(jnp.asarray(data).reshape(-1, 1), equi=True)
        assert float((g - cj.chebfun(expression)).norm()) < 10 * _vscale(g) * EPS

    def test_scalar_constant(self):
        # pass(10): a single scalar -> constant Chebfun.
        u = float(XX[-1])
        g = cj.chebfun(u, equi=True)
        assert float((g - u).norm()) < 10 * _vscale(g) * EPS

    def test_equi_with_function_handle_errors(self):
        # pass(11): 'equi' with a function handle (adaptive) is an error.
        with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:parseInputs:equi"):
            cj.chebfun("exp(x)", equi=True)
