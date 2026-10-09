"""Literal whichInterval endpoint branches for native C1 functionals.

Provenance
----------
MATLAB source: @opDiscretization/whichInterval.m, @chebcolloc/feval.m,
    @functionalBlock/functionalBlock.m (feval sanity check).
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.operators._native_values import FirstKindDisc
from chebfunjax.operators.blocks import eval_at


def apply(domain, sizes, location, direction, values):
    block = eval_at(location, domain, direction)
    return block._values_capability.realize(FirstKindDisc(sizes, domain)) @ values


@pytest.mark.parametrize('location', [-1., 0., 2.])
@pytest.mark.parametrize('direction', [-1, 0, 1])
def test_single_interval_source_shortcut(location, direction):
    domain = (-1., 2.)
    disc = FirstKindDisc((8,), domain)
    result = apply(domain, (8,), location, direction, 1+2j*disc.points)
    assert jnp.abs(result-(1+2j*location)) < 1e-12


@pytest.mark.parametrize('location,direction,expected', [
    (-1., 0, 1+2j), (-1., 1, 1+2j),
    (2., 0, 3-1j), (2., -1, 3-1j),
    (0., -1, 1+2j), (0., 0, 3-1j), (0., 1, 3-1j),
])
def test_piecewise_endpoint_and_interior_directions(location, direction, expected):
    values = jnp.concatenate([jnp.full(6, 1+2j), jnp.full(8, 3-1j)])
    result = apply((-1., 0., 2.), (6, 8), location, direction, values)
    assert jnp.abs(result-expected) < 1e-12


def test_piecewise_right_outward_matches_native_error():
    with pytest.raises(ValueError) as error:
        apply((-1., 0., 2.), (6, 8), 2., 1, jnp.ones(14))
    assert str(error.value) == (
        'CHEBFUN:OPDISCRETIZATION:whichInterval:undefined -- '
        'Evaluation direction is undefined at the location.')


def test_piecewise_left_outward_does_not_wrap_python_index():
    # Native whichInterval returns MATLAB index0; feval then fails indexing.
    with pytest.raises(IndexError, match='before the domain'):
        apply((-1., 0., 2.), (6, 8), -1., -1, jnp.ones(14))


@pytest.mark.parametrize('location', [-2., 3.])
def test_outside_location_is_rejected_by_functional_factory(location):
    with pytest.raises(ValueError, match='outside domain'):
        eval_at(location, (-1., 0., 2.))
