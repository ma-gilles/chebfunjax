"""Original tests/chebfun/test_or.m, Chebfun7574c77, all12source slots."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj


def samples(key):
    with np.load(Path(__file__).parent / "fixtures/or_logical_rng6178.npz") as data:
        return jnp.asarray(data["or_" + key])


def test_source_slots_1_through_7():
    x = samples("x1")
    f = cj.chebfun(jnp.sin)
    assert (f | cj.chebfun()).isempty() and (cj.chebfun() | f).isempty()  # 1
    f = cj.chebfun(jnp.sin, domain=[-1, -0.5, 0.5, 1])
    g = cj.chebfun(lambda x: 0 * x)
    h = f | g
    indices = jnp.where(h.point_values == 0)[0]
    assert (
        bool(jnp.all(h(x) == 1))
        and indices.size == 1
        and bool(
            jnp.all(
                jnp.abs(jnp.asarray(h.domain.breakpoints)[indices])
                < 10 * h.vscale * jnp.finfo(jnp.float64).eps
            )
        )
    )  # 2
    g = cj.chebfun(jnp.exp)
    h = f | g
    assert bool(jnp.array_equal(h.point_values, jnp.asarray([1, 1]))) and bool(
        jnp.all(h(x) == 1)
    )  # 3
    f = cj.chebfun(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
    g = cj.chebfun(lambda x: jnp.stack([0 * x, jnp.exp(x)], axis=-1))
    expected = jnp.stack([jnp.logical_or(jnp.sin(x), 0 * x), 0 * x + 1], axis=-1)
    h = f | g
    error = h(x) - expected
    assert float(jnp.max(jnp.abs(error.reshape(-1)))) == 0 and bool(
        jnp.array_equal(h.point_values, jnp.asarray([[1, 1], [0, 1], [1, 1]]))
    )  # 4
    h = f.transpose() | g.transpose()
    error = h(x) - expected.T
    assert float(jnp.max(jnp.abs(error.reshape(-1)))) == 0 and bool(
        jnp.array_equal(h.point_values, jnp.asarray([[1, 1], [0, 1], [1, 1]]))
    )  # 5
    f = cj.chebfun(jnp.sin)
    g = cj.chebfun(lambda x: jnp.stack([jnp.sin(x), jnp.exp(x)], axis=-1))
    with pytest.raises(ValueError, match="^CHEBFUN:CHEBFUN:or:dims:"):
        f | g  # 6
    g = cj.chebfun(jnp.cos, domain=[-2, 7])
    with pytest.raises(ValueError, match="^CHEBFUN:CHEBFUN:or:doms:"):
        f | g  # 7


def test_source_slots_8_through_11():
    dom = [-2, 7]
    f = cj.chebfun(
        lambda x: jnp.sin(30 * x) / ((x - dom[0]) * (x - dom[1])),
        domain=dom,
        exps=[-1, -1],
        splitting=True,
    )
    g = cj.chebfun(lambda x: 0 * x, domain=dom, splitting=True)
    h1 = f | f
    h2 = f | g
    x = samples("x2")
    assert not bool(jnp.any(h1(x) - 1))  # 8
    roots = f.roots()
    assert not bool(jnp.any(h1(roots)))  # 9
    assert not bool(jnp.any(h2(x) - 1))  # 10
    roots = f.roots()
    assert not bool(jnp.any(h2(roots)))  # 11


def test_source_slot_12():
    dom = [-jnp.inf, 0, 3 * jnp.pi]
    x = samples("x3")
    op = lambda x: 5 + jnp.exp(x**3)
    f = cj.chebfun([op, 0], domain=dom)
    _g = cj.chebfun(lambda x: 0 * x, domain=[dom[0], dom[2]])  # Literal unused native construction.
    h = f.logical()  # Original OR file deliberately tests LOGICAL in slot12.
    error = h(x) - (op(x) != 0)
    assert not bool(jnp.any(error)) and bool(
        jnp.array_equal(h.point_values, jnp.asarray([1, 1, 0]))
    )


class TestChebfunOr:
    def test_or_indicator(self):
        f = cj.chebfun(jnp.exp)
        z = cj.chebfun(lambda x: jnp.zeros_like(x))
        assert float(f.logical_or(z)(jnp.asarray(0.2))) == 1.0
        assert float(z.logical_or(z)(jnp.asarray(0.2))) == 0.0
