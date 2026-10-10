"""All ten assertions from MATLAB Chebfun tests/chebfun3/test_plus.m.

Chebfun commit: 7574c77. Original functions, domains, construction sequence,
continuous norm predicates, rank comparisons and thresholds are retained.
MATLAB single-output rank is max of the Python Tucker-rank tuple.
Resource qualification may split cases into fresh serial processes; this
changes cache lifetime, not the computations within each assertion block.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref

TOL = 1e5 * ChebfunPref().cheb3Prefs.chebfun3eps
DOMAINS = [
    (-1., 1., -1., 1., -1., 1.),
    (-2., 2., -2., 2., -2., 2.),
    (-1., math.pi, 0., 2*math.pi, -math.pi, math.pi),
]


def ff(x, y, z):
    return jnp.cos(x*y*z)


def gg(x, y, z):
    return x+y+z+x*y*z


@pytest.mark.parametrize("statement", [1, 2, 3])
def test_native_plus_domain_norm(statement):
    domain = DOMAINS[statement-1]
    f = chebfun3(ff, domain)
    g = chebfun3(gg, domain)
    exact = chebfun3(lambda x, y, z: ff(x, y, z)+gg(x, y, z), domain)
    tolk = max(abs(endpoint) for endpoint in domain)*TOL
    assert (f+g-exact).norm() < tolk


@pytest.mark.parametrize("statement", [4, 5])
def test_native_plus_rank(statement):
    f = chebfun3((lambda x, y, z: x) if statement == 4 else ff)
    g = f+f
    if statement == 4:
        assert max(g.rank) == max(f.rank)
    else:
        assert max(g.rank) <= max(f.rank)


def test_native_plus_small_scale():
    f = chebfun3(lambda x, y, z: 1e-10*x)
    g = f+f
    assert max(g.rank) == max(f.rank)  # Native pass(6).
    assert abs(g.vscale()-2e-10) < TOL  # Native pass(7).


def test_native_plus_large_scale():
    f = chebfun3(lambda x, y, z: 1e100*x)
    g = f+f
    assert max(g.rank) == max(f.rank)  # Native pass(8).
    assert abs(g.vscale()-2e100)/2e100 < 2e-2  # Native pass(9).


def test_native_plus_mixed_tech():
    def operator(x, y, z):
        return jnp.sin(math.pi*x)*jnp.cos(math.pi*(x+y))

    f = chebfun3(operator)
    g = chebfun3(operator, trig=True)
    difference = f-g
    assert difference.norm() < TOL  # Native pass(10).
