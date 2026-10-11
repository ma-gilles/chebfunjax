"""Literal eleven predicates from MATLAB's source test.

Python adapters: trigpts/legpts return (nodes, weights), max/min return
(location, value), and return_constant replaces MATLAB's second output.

Provenance
----------
MATLAB source: tests/misc/test_lebesgue.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp

from chebfunjax import lebesgue
from chebfunjax.utils.quadrature import chebpts, legpts, trigpts

EPS = float(jnp.finfo(jnp.float64).eps)


def test_polynomial_1():
    L, C = lebesgue(chebpts(3), return_constant=True)
    assert abs(C - 5 / 4) < 10 * EPS


def test_polynomial_2():
    L, C = lebesgue(legpts(3)[0], [-1, 1], return_constant=True)
    assert abs(C - 7 / 3) < 11 * EPS


def test_polynomial_3():
    L, C = lebesgue(jnp.linspace(5, 9, 3), 5, 9, return_constant=True)
    assert abs(C - 5 / 4) < 10 * EPS


def test_polynomial_4():
    L = lebesgue([1, 2], [0, 7])
    assert abs(L.norm(jnp.inf) - 11) < 100 * EPS


def test_polynomial_5():
    L, C = lebesgue([-1, 0, 1], [-1, 2], return_constant=True)
    assert jnp.max(jnp.abs(jnp.array([7, 7]) - jnp.array([C, L.max()[1]]))) < 100 * EPS


def test_polynomial_6():
    s = jnp.linspace(0.25, 1, 17)
    L, C = lebesgue(jnp.concatenate((-s, s)), return_constant=True)
    assert L.min()[1] > 0.999


def test_polynomial_7():
    L = lebesgue(jnp.linspace(-1, 1, 40))
    assert len(L) < 1560


def test_trig_8():
    L, C = lebesgue(trigpts(3)[0], "trig", return_constant=True)
    assert abs(C - 5 / 3) < 10 * EPS


def test_trig_9():
    L, C = lebesgue(
        jnp.linspace(-jnp.pi, jnp.pi, 4)[:-1], [-jnp.pi, jnp.pi], "trig", return_constant=True
    )
    assert abs(C - 5 / 3) < 10 * EPS


def test_trig_10():
    L, C = lebesgue(jnp.linspace(-jnp.pi, jnp.pi, 4), -jnp.pi, jnp.pi, "trig", return_constant=True)
    assert abs(C - 5 / 3) < 10 * EPS


def test_trig_11():
    try:
        lebesgue(trigpts(4)[0], "trig")
    except ValueError as e:
        assert str(e) == "CHEBFUN:lebesgue:trigLebesgue:evenLengthGrid"
        return
    raise AssertionError("missing evenLengthGrid error")
