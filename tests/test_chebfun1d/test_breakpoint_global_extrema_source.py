"""Native tests/chebfun/test_max.m pass6 and isolated point regressions."""

import jax.numpy as jnp
import pytest

from chebfunjax import chebfun


def test_native_impulse_point_values():
    f = chebfun(
        [lambda x: -jnp.ones_like(x), lambda x: jnp.ones_like(x), lambda x: 2 * jnp.ones_like(x)],
        domain=[-1.0, 0.0, 1.0, 2.0],
    )
    assert f.max()[1] == 2
    values = f.point_values.at[0].set(10.0).at[2].set(-10.0)
    f = f.set_point_values(values)
    assert f.max() == (-1.0, 10.0)
    assert f.min() == (1.0, -10.0)


@pytest.mark.parametrize(
    "point_values, expected_min, expected_max",
    [
        ([-4.0, 3.0], (-1.0, -4.0), (1.0, 3.0)),
        ([4.0, -3.0], (1.0, -3.0), (-1.0, 4.0)),
        ([0.0, 0.0], (-1.0, 0.0), (-1.0, 0.0)),
    ],
)
def test_explicit_breakpoint_extrema(point_values, expected_min, expected_max):
    f = chebfun(lambda x: jnp.zeros_like(x)).set_point_values(jnp.asarray(point_values))
    assert f.minandmax() == (expected_min, expected_max)
