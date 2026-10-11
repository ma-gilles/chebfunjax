"""CPU prefix adapter controls; native root tolerances remain unchanged."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech import chebtech as module


@pytest.mark.parametrize("n", [51, 52, 53, 54, 55, 56, 57, 58, 60, 64, 513])
def test_cpu_prefix_basis_and_zero_storage(n):
    left, right = module._roots_cached_transforms(jax.devices("cpu")[0], True)
    for index in (0, n - 4, n - 1):
        coefficients = np.zeros(n)
        coefficients[index] = 1.0
        actual = module._roots_subdivide(coefficients)
        for child, matrix in zip(actual, (left, right)):
            np.testing.assert_array_equal(child, np.asarray(matrix)[:n, index])
            assert child.shape == (n,) and child.flags.owndata
    for child in module._roots_subdivide(np.zeros(n)):
        assert np.array_equal(child.view(np.uint64), np.zeros(n).view(np.uint64))


@pytest.mark.parametrize("kind", ["complex", "float32", "device", "small", "matrix"])
def test_ineligible_inputs_keep_generic_path(monkeypatch, kind):
    coefficients = np.arange(60, dtype=float) / 60
    if kind == "complex":
        coefficients = coefficients.astype(complex) * (1 + 2j)
    elif kind == "float32":
        coefficients = coefficients.astype(np.float32)
    elif kind == "matrix":
        coefficients = np.stack((coefficients, -coefficients), axis=1)
    elif kind == "device":
        coefficients = jnp.asarray(coefficients)
    else:
        coefficients = coefficients[:50]

    def forbidden(*args):
        raise AssertionError("ineligible input reached CPU lane adapter")

    monkeypatch.setattr(module, "_roots_apply_transforms_cpu_lanes", forbidden)
    left, right = module._roots_subdivide(coefficients)
    assert left.shape == right.shape == coefficients.shape


def test_nonfinite_host_coefficients_return_to_generic_path(monkeypatch):
    coefficients = np.zeros(60)
    coefficients[4] = np.nan
    calls = []

    def generic(c, left, right):
        calls.append(np.asarray(c))
        return "left fallback", "right fallback"

    monkeypatch.setattr(module, "_roots_apply_transforms", generic)
    assert module._roots_subdivide(coefficients) == ("left fallback", "right fallback")
    assert len(calls) == 1 and np.isnan(calls[0][4])


def test_generic_kernel_remains_jit_and_forward_differentiable():
    left, right = module._roots_cached_transforms(jax.devices("cpu")[0], True)
    c = jnp.arange(60.0) / 60
    tangent = jnp.eye(60)[3]

    def fn(values):
        return module._roots_apply_transforms(values, left, right)

    _, derivative = jax.jvp(jax.jit(fn), (c,), (tangent,))
    for actual, matrix in zip(derivative, (left, right)):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(matrix)[:60, 3])


@pytest.mark.parametrize("n", [60, 51, 53, 54])
def test_retained_multiterm_cpu_transform_results(n):
    from pathlib import Path

    fixture = Path(__file__).parents[1] / "fixtures/roots_cpu_lane_actual_inputs.npz"
    with np.load(fixture) as data:
        actual = module._roots_subdivide(data[f"n{n}_input"])
        for child, name in zip(actual, ("left", "right")):
            np.testing.assert_array_equal(child, data[f"n{n}_{name}"])
