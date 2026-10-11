"""Source composition validation and periodic numeric controls, pin7574c77."""

import gc

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebfun3d.chebfun3 import Chebfun3


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    jax.clear_caches()
    gc.collect()


def test_two_component_domain_rejection():
    f = Chebfun2v.from_functions(lambda x, y: 2 * x, lambda x, y: y)
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN2V:COMPOSE:DomainMismatch2"):
        chebfun2(lambda x, y: x + y)(f)


def test_complex_rejected_before_wrong_operand():
    f = Chebfun2v.from_functions(lambda x, y: 1j * x, lambda x, y: y)
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN2V:COMPOSE:Complex"):
        f.compose(lambda x, y: x + y)


@pytest.mark.parametrize("count", [2, 3])
def test_operand_dimension_errors(count):
    f = Chebfun2v.from_functions(*([lambda x, y: x] * count))
    g = (
        Chebfun3.from_function(lambda x, y, z: x + y + z)
        if count == 2
        else chebfun2(lambda x, y: x + y)
    )
    with pytest.raises(ValueError, match=f"CHEBFUN:CHEBFUN2V:COMPOSE:OP{count}"):
        f.compose(g)


def test_periodic_composition_values():
    a = chebfun2(lambda x, y: jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y), trig=True)
    b = chebfun2(lambda x, y: jnp.cos(jnp.pi * x) * jnp.cos(jnp.pi * y), trig=True)
    f = Chebfun2v([a.approx, b.approx])
    g = Chebfun2v.from_functions(lambda x, y: x * x + y * y, lambda x, y: x)
    h = g(f)
    points = jnp.linspace(-1.0, 1.0, 17)
    x, y = jnp.meshgrid(points, points)
    truth = jnp.stack([jnp.cos(jnp.pi * x) ** 2, a(x, y)], axis=-1)
    assert jnp.max(jnp.abs(h(x, y) - truth)) < 1000 * jnp.finfo(jnp.float64).eps
