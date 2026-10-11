"""Original tests/chebfun/test_logical.m, Chebfun7574c77, all9source slots."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


def samples(key):
    with np.load(Path(__file__).parent / "fixtures/or_logical_rng6178.npz") as data:
        return jnp.asarray(data["logical_" + key])


def test_source_slots_1_through_6():
    x = samples("x1")
    f = cj.chebfun(lambda x: jnp.sin(2 * x), domain=[-1, 1])
    g = f.logical()
    assert bool(jnp.array_equal(g.point_values, jnp.asarray([1, 0, 1]))) and bool(
        jnp.all(g(x) == 1)
    )  # 1
    f = cj.chebfun(jnp.exp, domain=[-1, -0.5, 0.5, 1])
    g = f.logical()
    assert bool(jnp.all(g.point_values == 1)) and bool(jnp.all(g(x) == 1))  # 2
    f = cj.chebfun(lambda x: 0 * x, domain=[-1, -0.5, 0.5, 1])
    g = f.logical()
    assert bool(jnp.all(g.point_values == 0)) and bool(jnp.all(g(x) == 0))  # 3
    f = cj.chebfun(lambda x: 0 * x + 1, domain=[-1, -0.5, 0.5, 1])
    g = f.logical()
    assert bool(jnp.all(g.point_values == 1)) and bool(jnp.all(g(x) == 1))  # 4
    f = cj.chebfun(lambda x: jnp.exp(1j * x) * jnp.sin(x), domain=[-1, -0.5, 0.5, 1])
    g = f.logical()
    assert bool(jnp.array_equal(g.point_values, jnp.asarray([1, 1, 0, 1, 1]))) and bool(
        jnp.all(g(x) == 1)
    )  # 5
    f = cj.chebfun(
        lambda x: jnp.stack([jnp.sin(x), jnp.exp(x)], axis=-1), domain=[-1, -0.5, 0.5, 1]
    )
    g = f.logical()
    assert bool(
        jnp.array_equal(g.point_values, jnp.asarray([[1, 1, 0, 1, 1], [1, 1, 1, 1, 1]]).T)
    ) and bool(jnp.all(g(x) == 1))  # 6


def test_source_slots_7_and_8():
    dom = [-2, 7]
    f = cj.chebfun(
        lambda x: jnp.sin(30 * x) / ((x - dom[0]) * (x - dom[1])),
        domain=dom,
        exps=[-1, -1],
        splitting=True,
    )
    h = f.logical()
    x = samples("x2")
    assert not bool(jnp.any(h(x) - 1))  # 7
    r = f.roots()
    assert not bool(jnp.any(h(r)))  # 8


def test_source_slot_9():
    dom = [-jnp.inf, 0, 3 * jnp.pi]
    x = samples("x3")
    op = lambda x: 5 + jnp.exp(x**3)
    f = cj.chebfun([op, 0], domain=dom)
    g = f.logical()
    error = g(x) - (op(x) != 0)
    assert not bool(jnp.any(error))  # 9


class TestChebfunLogical:
    def test_logical_of_nonzero(self):
        f = cj.chebfun(jnp.exp)
        ind = f.logical_ne(0.0)
        assert float(ind(jnp.asarray(0.4))) == 1.0
