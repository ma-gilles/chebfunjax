"""Actual native RHS states and independent licensed BLAS reference outputs."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d._pde_ndf.pde_driver import Values
from chebfunjax.chebfun1d._pde_ndf.source_arithmetic import binary, matvec
from chebfunjax.utils.diffmat import diffmat

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu" or not jax.config.x64_enabled,
    reason="Native arithmetic qualification is CPU binary64",
)
FIXTURE = Path(__file__).parents[1] / "references/pde_cpu_arithmetic_native.npz"


def bits(actual, expected):
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint64), np.asarray(expected).view(np.uint64)
    )


@pytest.mark.parametrize("n", [2, 3, 4, 7, 8, 9, 17, 32, 33, 65, 129, 130, 258])
def test_native_gemv(n):
    with np.load(FIXTURE) as data:
        a, x, expected = (data[f"gemv_{n}_{key}"] for key in ["matrix", "input", "expected"])
    bits(matvec(jnp.asarray(a), jnp.asarray(x)), expected)


@pytest.mark.parametrize(
    "case,query",
    [
        (0, 0),
        (0, 1),
        (0, 33),
        (1, 0),
        (1, 1),
        (1, 65),
        (2, 0),
        (2, 1),
        (2, 33),
        (3, 0),
        (3, 1),
        (3, 65),
    ],
)
def test_public_values_rhs_on_actual_native_state(case, query):
    with np.load(FIXTURE) as data:
        y, expected = data[f"rhs_{case}_{query}_input"], data[f"rhs_{case}_{query}_expected"]
        projection, boundary, q, offset, domain = (
            jnp.asarray(data[f"case_{case}_{key}"])
            for key in ["P", "B", "q", "BCVALOFFSET", "domain"]
        )
    n = len(y)
    scale = 2.0 / (domain[-1] - domain[0])
    d1, d2 = diffmat(n, p=1), diffmat(n, p=2)

    @jax.jit
    def rhs(z):
        u = Values(z, {"D": {1: d1, 2: d2}, "scale": scale, "order": 0})
        value = matvec(projection, (0.1 * u.diff(2) + u.diff()).vals)
        value = value.at[0].set((boundary.reshape(1, n) @ z)[0] - jnp.atleast_1d(q)[0])
        value = value.at[1].set(u.diff().vals[0])
        return value.at[:2].add(-offset)

    bits(rhs(jnp.asarray(y)), expected)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex128])
def test_generic_dtype_fallback(dtype):
    a = jnp.arange(12, dtype=jnp.float64).reshape(3, 4).astype(dtype) / 7
    x = jnp.asarray([1.0, -2.0, 3.0, -4.0], dtype=dtype)
    np.testing.assert_array_equal(matvec(a, x), a @ x)


@pytest.mark.parametrize("shape", [(1, 33), (515, 2), (2, 515)])
def test_generic_shape_fallback(shape):
    a = jnp.arange(shape[0] * shape[1], dtype=jnp.float64).reshape(shape) / 13
    x = jnp.linspace(-1.0, 1.0, shape[1])
    bits(matvec(a, x), a @ x)


def test_nonfinite_fallback_and_ad():
    a = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x = jnp.asarray([1.0, -2.0, 3.0])
    np.testing.assert_array_equal(jax.jacfwd(lambda z: matvec(a, z))(x), a)
    bad = a.at[0, 1].set(jnp.inf)
    np.testing.assert_array_equal(matvec(bad, x), bad @ x)
    np.testing.assert_array_equal(
        binary(jnp.asarray([jnp.inf]), jnp.asarray([0.0]), 0), jnp.asarray([jnp.nan])
    )


def test_binary_signed_zero_and_transverse_derivative():
    x = jnp.asarray([-0.0, 0.0, -2.0, 3.0])
    one = jnp.ones_like(x)
    bits(binary(x, one, 0), x)
    np.testing.assert_array_equal(jax.jacfwd(lambda z: binary(z, one, 0))(x), jnp.eye(4))
