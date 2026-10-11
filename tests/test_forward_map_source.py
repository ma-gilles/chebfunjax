"""Source-rounded affine maps, including eager/JIT and identity contracts."""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.domain import Domain
from chebfunjax.utils.quadrature import chebpts, chebpts_ab


@pytest.mark.parametrize('a,b',[(-2.,7.),(-2.5,3.),(-7.3,-2.1),(0.,7.)])
def test_source_arithmetic_eager_and_traced(a,b):
    t=chebpts(65)
    expected=jnp.asarray([b*(float(v)+1)/2+a*(1-float(v))/2 for v in t])
    d=Domain((a,b))
    assert bool(jnp.array_equal(d.forward_map(t),expected))
    assert bool(jnp.array_equal(jax.jit(d.forward_map)(t),expected))
    assert bool(jnp.array_equal(chebpts_ab(65,a,b),expected))
    assert bool(jnp.array_equal(jax.jit(lambda a,b:chebpts_ab(65,a,b))(a,b),expected))
    derivative=jax.grad(lambda y:d.forward_map(y))(jnp.asarray(.31))
    assert abs(float(derivative)-(b-a)/2)<2e-15


def test_reference_interval_is_bitwise_identity():
    t=chebpts(65)
    d=Domain((-1.,1.))
    assert bool(jnp.array_equal(d.forward_map(t),t))
    assert bool(jnp.array_equal(jax.jit(d.forward_map)(t),t))
    assert bool(jnp.array_equal(jax.jit(lambda a,b:chebpts_ab(65,a,b))(-1.,1.),t))
