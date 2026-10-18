"""Analytic derivatives of source-rounded cosine, including constant branches."""

import json
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.utils._fdlibm_boundary_cosine import _cos_boundary_angles


@pytest.mark.parametrize(
    "point",
    [
        0.0,
        -0.0,
        2.0**-30,
        -(2.0**-30),
        0.7730104533627369,
        -0.7730104533627369,
        jnp.pi / 2,
        -jnp.pi / 2,
        jnp.pi,
        -jnp.pi,
    ],
)
def test_first_four_analytic_derivatives(point):
    x = jnp.asarray(point, jnp.float64)
    derivative = _cos_boundary_angles
    expected = (-jnp.sin(x), -jnp.cos(x), jnp.sin(x), jnp.cos(x))
    for target in expected:
        previous = derivative

        def derivative(z, previous=previous):
            return jax.jvp(previous, (z,), (jnp.ones_like(z),))[1]

        actual = derivative(x)
        assert bool(jnp.abs(actual - target) <= 4 * jnp.finfo(jnp.float64).eps)


def test_reverse_over_forward_array_hessian():
    x = jnp.asarray([[0.0, 2.0**-30], [0.7730104533627369, jnp.pi]])
    weights = jnp.asarray([[0.25, -0.5], [1.5, 2.0]])
    hessian = jax.jacfwd(jax.grad(lambda a: jnp.sum(weights * _cos_boundary_angles(a))))(x)
    expected = jnp.diag((-weights * jnp.cos(x)).reshape(-1)).reshape(x.shape + x.shape)
    assert bool(jnp.max(jnp.abs(hessian - expected)) <= 4 * jnp.finfo(jnp.float64).eps)


def test_float32_input_cast_retains_derivative_dtype():
    x = jnp.asarray([0.0, 2.0**-30, 0.5], jnp.float32)
    primal, tangent = jax.jvp(_cos_boundary_angles, (x,), (jnp.ones_like(x),))
    gradient = jax.grad(lambda a: jnp.sum(_cos_boundary_angles(a)))(x)
    assert primal.dtype == tangent.dtype == jnp.float64
    assert gradient.dtype == jnp.float32
    assert bool(
        jnp.max(jnp.abs(tangent + jnp.sin(x.astype(jnp.float64)))) <= 4 * jnp.finfo(jnp.float64).eps
    )


@pytest.mark.parametrize("row_index", [0, 1, 2], ids=["independent1040", "actual17", "actual33"])
def test_primal_retains_native_words(row_index):
    fixture = Path(__file__).parent / "fixtures/bounded_cosine_native_20261017.json"
    row = json.loads(fixture.read_text())["rows"][row_index]
    values = jnp.asarray(
        [struct.unpack(">d", bytes.fromhex(word))[0] for word in row["inputs"]], jnp.float64
    )
    actual = _cos_boundary_angles(values)
    assert [struct.pack(">d", value).hex() for value in actual.tolist()] == row["expected"]
