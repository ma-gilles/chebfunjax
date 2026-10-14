"""Captured native ACA3 arithmetic, Chebfun 7574c77 / MATLAB R2025b."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d._chebfun3double import _subtract3, _subtract_outer
from chebfunjax.utils._ordered_binary import binary


@pytest.mark.parametrize("outer_jit", [False, True])
def test_native_14_post_gemm_steps(outer_jit):
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        pytest.skip("Native rounding qualification is CPU binary64")
    update = jax.jit(lambda a, m, f, p: _subtract_outer(a, m, binary(f, p, 3)))
    if outer_jit:
        update = jax.jit(update)
    with np.load(Path(__file__).parent / "fixtures/aca3_native_intermediates.npz") as data:
        for k in range(14):
            a, m, f, p = [
                jnp.asarray(data[f"{name}_{k}"]) for name in ("tensor", "matrix", "fiber", "pivot")
            ]
            np.testing.assert_array_equal(
                np.asarray(binary(f, p, 3)).view(np.uint64), data[f"vector_{k}"].view(np.uint64)
            )
            np.testing.assert_array_equal(
                np.asarray(update(a, m, f, p)).view(np.uint64),
                data[f"expected_{k}"].view(np.uint64),
            )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex128])
def test_generic_dtype(dtype):
    a = jnp.ones((2, 2, 2), dtype=dtype)
    c = jnp.asarray([[2.0], [4.0]], dtype=dtype)
    r = jnp.asarray([[3.0], [5.0]], dtype=dtype)
    out = _subtract3(
        a,
        c,
        jnp.asarray([2.0], dtype=dtype),
        r,
        jnp.asarray([2.0, 4.0], dtype=dtype),
        jnp.asarray(2.0, dtype=dtype),
    )
    np.testing.assert_array_equal(
        out, np.array([[[-2.0, -5.0], [-4.0, -9.0]], [[-5.0, -11.0], [-9.0, -19.0]]])
    )


def test_signed_zero_outer():
    a = np.array([[[0.0, -0.0]], [[-0.0, 0.0]]])
    m = np.array([[0.0], [-0.0]])
    v = np.array([1.0, -1.0])
    expected = a - (m[..., None] * v)
    actual = jax.jit(_subtract_outer)(jnp.asarray(a), jnp.asarray(m), jnp.asarray(v))
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint64), expected.view(np.uint64))


def test_fiber_derivative():
    a = jnp.ones((2, 2, 2))
    c = jnp.array([[2.0], [4.0]])
    r = jnp.array([[3.0], [5.0]])

    def fun(f):
        return _subtract3(a, c, jnp.array([2.0]), r, f, jnp.array(2.0))

    _, derivative = jax.jvp(fun, (jnp.array([2.0, 4.0]),), (jnp.array([1.0, 2.0]),))
    np.testing.assert_array_equal(
        derivative, -np.array([[3.0, 5.0], [6.0, 10.0]])[..., None] * np.array([0.5, 1.0])
    )


@pytest.mark.parametrize("value", [float("inf"), float("nan"), 0.0])
def test_nonfinite_zero_pivot_fallback(value):
    a = jnp.ones((1, 1, 2))
    c = jnp.ones((1, 1))
    p = jnp.ones(1)
    f = jnp.array([1.0, 2.0])
    pivot = jnp.asarray(value)
    out = np.asarray(_subtract3(a, c, p, c, f, pivot))
    with np.errstate(all="ignore"):
        expected = np.ones((1, 1, 2)) - np.array([1.0, 2.0]) / value
    np.testing.assert_array_equal(out, expected)
