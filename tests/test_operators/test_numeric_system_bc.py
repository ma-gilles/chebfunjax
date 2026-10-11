"""Native parseBC numeric vector contracts; Chebfun commit 7574c77."""

import jax.numpy as jnp
import pytest

from chebfunjax.operators.chebop import _as_system_bc


@pytest.mark.parametrize(
    "values",
    [
        [0.0, 1.0],
        [jnp.sin(0.0), jnp.cos(0.0)],
        jnp.array([0.0, 1.0]),
        jnp.array([[0.0], [1.0]]),
        jnp.array([[0.0, 1.0]]),
    ],
)
def test_numeric_system_vectors(values):
    result = _as_system_bc(values, 2)(jnp.array([2.0, 3.0]), jnp.array([4.0, 5.0]))
    assert jnp.array_equal(result[0], jnp.array([2.0, 3.0]))
    assert jnp.array_equal(result[1], jnp.array([3.0, 4.0]))


@pytest.mark.parametrize("values", [[1.0], [1.0, 2.0, 3.0], jnp.array(1.0)])
def test_number_of_conditions(values):
    with pytest.raises(ValueError, match="CHEBFUN:CHEBOP:parseBC:numberOfConditions"):
        _as_system_bc(values, 2)


def test_callable_and_scalar_system_distinction():
    def condition(u, v):
        return [u + v - 1, u - v]

    assert _as_system_bc(condition, 2) is condition
    derivatives = [1.0, 2.0]
    assert _as_system_bc(derivatives, 1) is derivatives
    assert _as_system_bc(None, 2) is None
    assert _as_system_bc(jnp.array([]), 2) is None


def test_complex_numeric_values_not_narrowed():
    condition = _as_system_bc(jnp.array([1 + 2j, 3 - 4j]), 2)
    values = condition(jnp.array(5 + 6j), jnp.array(7 + 8j))
    assert jnp.array_equal(jnp.asarray(values), jnp.array([4 + 4j, 4 + 12j]))
