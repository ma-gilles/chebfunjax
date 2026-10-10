"""Native cubic basis, exact edge stencils, and generic JAX transformations."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.utils._interp2_cubic import uniform_cubic_interp2

F = json.loads(Path(__file__).with_name("interp2_cubic_native_r2025b.json").read_text())


def array(values):
    # Native outside-grid NaNs are stored as JSON null.
    if isinstance(values, list):
        return [array(v) for v in values]
    return float("nan") if values is None else values


def native_query(values):
    return uniform_cubic_interp2(values, jnp.asarray(F["xq"]),
                                 jnp.asarray(F["yq"])[:, None],
                                 x_domain=(0., 4.), y_domain=(0., 4.))


@pytest.mark.parametrize("compiled", [False, True])
def test_native_all_25_basis_stencils(compiled):
    # MATLAB unit matrices use column-major indexing.
    basis = jnp.eye(25, dtype=jnp.float64).reshape(25, 5, 5).swapaxes(1, 2)
    evaluate = jax.vmap(native_query)
    if compiled:
        evaluate = jax.jit(evaluate)
    actual = evaluate(basis).transpose(1, 2, 0)
    expected = jnp.asarray(array(F["weights"]), dtype=jnp.float64)
    assert bool(jnp.array_equal(jnp.isnan(actual), jnp.isnan(expected)))
    mask = jnp.isfinite(expected)
    assert bool(jnp.array_equal(actual[mask].view(jnp.uint64), expected[mask].view(jnp.uint64)))


@pytest.mark.parametrize("compiled", [False, True])
def test_native_mixed_values(compiled):
    evaluate = jax.jit(native_query) if compiled else native_query
    actual = evaluate(jnp.asarray(F["v"]))
    expected = jnp.asarray(array(F["cubic_values"]))
    assert bool(jnp.array_equal(jnp.isnan(actual), jnp.isnan(expected)))
    # Four-term stencils in each axis plus quadratic edge extension; this
    # absolute bound includes 64 binary64 rounding units at input scale.
    bound = 64*jnp.finfo(jnp.float64).eps*jnp.max(jnp.abs(jnp.asarray(F["v"])))
    assert bool(jnp.all(jnp.abs(actual[jnp.isfinite(expected)]-expected[jnp.isfinite(expected)]) <= bound))


@pytest.mark.parametrize("complex_values", [False, True])
def test_translated_rectangular_quadratic_and_value_derivative(complex_values):
    x, y = jnp.linspace(-2., 2., 7), jnp.linspace(-.5, 0., 4)
    def polynomial(x, y):
        return x*x+2*y*y+x*y+1
    values = polynomial(x[None, :], y[:, None])
    factor = 1.+2j if complex_values else 1.
    values = factor*values
    xq, yq = jnp.linspace(-2., 2., 13), jnp.linspace(-.5, 0., 9)[:, None]
    def evaluate(v):
        return uniform_cubic_interp2(v, xq, yq, x_domain=(-2., 2.), y_domain=(-.5, 0.))
    actual = jax.jit(evaluate)(values)
    expected = factor*polynomial(xq, yq)
    bound = 64*jnp.finfo(jnp.float64).eps*jnp.max(jnp.abs(values))
    assert bool(jnp.all(jnp.abs(actual-expected) <= bound))
    _, tangent = jax.jvp(jax.jit(evaluate), (values,), (jnp.ones_like(values),))
    assert bool(jnp.all(jnp.abs(tangent-1) <= 64*jnp.finfo(jnp.float64).eps))


def test_query_nan_and_shape_validation():
    values = jnp.ones((4, 4))
    out = uniform_cubic_interp2(values, jnp.asarray([jnp.nan, jnp.inf, -jnp.inf, -1., 5.]),
                                0., x_domain=(0., 4.), y_domain=(0., 4.))
    assert bool(jnp.all(jnp.isnan(out)))
    with pytest.raises(ValueError, match="at least 4"):
        uniform_cubic_interp2(values[:3], 0., 0., x_domain=(0., 4.), y_domain=(0., 4.))
