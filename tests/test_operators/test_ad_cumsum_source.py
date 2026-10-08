"""Bounded cumsum cases from the original AD integration Taylor test.

Provenance: tests/adchebfun/test_cumsumDiffSumMean.m and
@adchebfun/adchebfun.m519-533; Chebfun 7574c77680d7e82b79626300bf255498271a72df.
The source tolerances are unchanged: first-order slope 1e-2, remainder 1e-12.
Fixed polynomial inputs replace source rand vectors; no RNG-stream claim.
"""
import math

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.blocks import ChebColloc2Disc


@pytest.mark.parametrize('domain', [(-1., 1.), (2., 5.), (-1., .2, 1.)])
@pytest.mark.parametrize('order', [0, 1, 2])
def test_constant_jacobian_anchoring(domain, order):
    """The derivative applied to one is (x-a)^k/k!, also across pieces."""
    u = chebfun(lambda x: .5 + .1*x*x, domain=domain)
    result = ADChebfun(u).cumsum(order)
    disc = ChebColloc2Disc(12, domain)
    points = disc.points()
    actual = result.jacobian.matrix(disc) @ jnp.ones(disc.n)
    expected = (points-domain[0])**order/math.factorial(order)
    assert float(jnp.max(jnp.abs(actual-expected))) < 1e-12
    assert result.domain == domain
    assert result.is_linear
    plain = u.cumsum(order)
    assert float(jnp.max(jnp.abs(result.func(points)-plain(points)))) == 0
    if order:
        assert float(jnp.abs(actual[0])) < 1e-12


@pytest.mark.parametrize('order', [1, 2])
def test_source_linear_taylor_controls(order):
    """Original h=5^-2..5^-5 sequence, slope and remainder bounds."""
    domain = (-1., 1.)
    u = chebfun(lambda x: .55+.03*x+.01*x*x, domain=domain)
    perturbation = chebfun(lambda x: .055+.003*x+.001*x*x, domain=domain)
    result = ADChebfun(u).cumsum(order)
    disc = ChebColloc2Disc(16, domain)
    x = disc.points()
    action = result.jacobian.matrix(disc) @ perturbation(x)
    first, second = [], []
    for h in [5.**-k for k in range(2, 6)]:
        delta = (u+h*perturbation).cumsum(order)(x)-result.func(x)
        first.append(jnp.max(jnp.abs(delta)))
        second.append(jnp.max(jnp.abs(delta-h*action)))
    slopes = jnp.diff(jnp.log(jnp.asarray(first)))/jnp.log(.2)
    assert float(jnp.max(jnp.abs(slopes-1))) < 1e-2
    assert float(jnp.max(jnp.asarray(second))) < 1e-12
    assert result.is_linear


def test_nonlinear_chain_rule_and_jax_action():
    """Integration retains nonlinear flags and differentiable JAX action."""
    domain = (-1., .1, 1.)
    u = chebfun(lambda x: 1+x, domain=domain)
    result = (ADChebfun(u)**2).cumsum(2)
    disc = ChebColloc2Disc(12, domain)
    x = disc.points()
    matrix = result.jacobian.matrix(disc)
    expected = (x+1)**3/3
    action = jax.jit(lambda v: matrix @ v)
    actual, tangent = jax.jvp(action, (jnp.ones(disc.n),), (jnp.ones(disc.n),))
    assert float(jnp.max(jnp.abs(actual-expected))) < 1e-12
    assert float(jnp.max(jnp.abs(tangent-expected))) < 1e-12
    assert not result.is_linear


@pytest.mark.parametrize('order', [-1, .5, float('inf'), float('nan')])
def test_invalid_source_operator_orders(order):
    u = ADChebfun(chebfun(lambda x: 1+x))
    with pytest.raises(ValueError, match='nonnegative integer'):
        u.cumsum(order)
