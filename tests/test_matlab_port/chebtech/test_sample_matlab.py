"""Port of MATLAB Chebfun tests/chebtech/test_sample.m (Opus 4.8).

MATLAB ``[v, p] = sample(f)`` / ``sample(f, m)`` returns the values ``v`` of the
chebtech on an ``m``-point Chebyshev grid together with the grid ``p``.
chebfunjax now implements ``sample`` (a port of ``@chebtech/sample.m``, which
aliases the coefficients to length ``m`` before converting to values), so the
tests run directly.

Provenance
----------
MATLAB source : tests/chebtech/test_sample.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.utils.quadrature import chebpts

CASES = [(Chebtech1, 1), (Chebtech2, 2)]

EPS = float(np.finfo(np.float64).eps)


def _n2(a):
    return float(jnp.linalg.norm(jnp.asarray(a)))


class TestChebtechSample:
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_grid_equal_length(self, Tech, kind):
        # pass(n, 1): sample(f) on a grid equal to length(f).
        f = Tech.from_function(lambda x: jnp.sin(x - 0.1))
        v, p = f.sample()
        p_ex = chebpts(len(f), kind)
        v_ex = f(p_ex)
        assert _n2(p - p_ex) < 100 * EPS
        assert _n2(v - v_ex) < 100 * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_grid_shorter(self, Tech, kind):
        # pass(n, 2): sample(f, m) with m < length(f).
        f = Tech.from_function(lambda x: jnp.sin(x - 0.1))
        m = round(len(f) / 2)
        v, p = f.sample(m)
        p_ex = chebpts(m, kind)
        v_ex = f(p_ex)
        assert _n2(p - p_ex) < 100 * EPS
        assert _n2(v - v_ex) < 100 * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_grid_longer(self, Tech, kind):
        # pass(n, 3): sample(f, m) with m > length(f).
        f = Tech.from_function(lambda x: jnp.sin(x - 0.1))
        m = round(2 * len(f))
        v, p = f.sample(m)
        p_ex = chebpts(m, kind)
        v_ex = f(p_ex)
        assert _n2(p - p_ex) < 100 * EPS
        assert _n2(v - v_ex) < 100 * EPS
