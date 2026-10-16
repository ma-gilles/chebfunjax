"""Source empty/error branches and independent numeric staging controls.

Provenance
----------
MATLAB source: @chebfun2v/subsref.m
Chebfun commit:7574c77
JIT/vmap/AD checks are Python controls, separate from original predicates.
"""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v


@pytest.fixture(scope="module")
def field():
    return Chebfun2v.from_functions(lambda x, y: x*x+y,
                                    lambda x, y: x-y*y)


def test_empty_index_precedes_index_validation():
    assert Chebfun2v.empty()[0].shape == (0,)
    assert Chebfun2v.empty()[99].shape == (0,)


@pytest.mark.parametrize("index", [-1, 2, 99])
def test_component_index_range(field, index):
    with pytest.raises(IndexError, match="CHEBFUN:CHEBFUN2V:subsref:index"):
        field[index]


@pytest.mark.parametrize("side", [0, 1])
def test_mixed_numeric_function_coordinates(field, side):
    coordinate = Chebfun2.from_function(lambda x, y: x)
    args = (coordinate, .25) if side == 0 else (.25, coordinate)
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN2V:subsref:input"):
        field(*args)


@pytest.mark.parametrize("mode", ["jit", "vmap", "grad"])
def test_numeric_evaluation_staging(field, mode):
    points = jnp.array([[-.75, .125], [.25, -.5], [.625, .75]])
    if mode == "grad":
        actual = jax.vmap(jax.grad(lambda p: jnp.sum(field(p[0], p[1]))))(points)
        expected = jnp.stack([2*points[:, 0]+1, 1-2*points[:, 1]], axis=-1)
    else:
        evaluate = (jax.jit(lambda p: field(p[:, 0], p[:, 1])) if mode == "jit"
                    else jax.vmap(lambda p: field(p[0], p[1])))
        actual = evaluate(points)
        expected = jnp.stack([points[:, 0]**2+points[:, 1],
                              points[:, 0]-points[:, 1]**2], axis=-1)
    assert jnp.max(jnp.abs(actual-expected)) < 100*jnp.finfo(jnp.float64).eps
