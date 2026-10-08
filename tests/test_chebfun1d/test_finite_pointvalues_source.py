"""Breakpoint handling in Chebfun7574c77 columnIsfinite/isinf."""

import jax.numpy as jnp
import pytest

import chebfunjax as cj


@pytest.mark.parametrize("value", [jnp.inf, -jnp.inf, jnp.nan])
def test_nonfinite_isolated_point(value):
    f = cj.chebfun(lambda x: x, domain=(-1, 0, 1))
    f = f.set_point_values(jnp.asarray([[-1.0], [value], [1.0]]))
    assert not f.isfinite()
    assert f.isinf()  # Literal source isinf is negated isfinite, even forNaN.


def test_finite_isolated_point():
    f = cj.chebfun(lambda x: x, domain=(-1, 0, 1))
    f = f.set_point_values(jnp.asarray([[-1.0], [1e300], [1.0]]))
    assert f.isfinite()
    assert not f.isinf()
