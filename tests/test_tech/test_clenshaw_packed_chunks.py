"""Chunk seams preserve22bbe4be register arithmetic and global IEEE choices."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import _clenshaw
from chebfunjax.utils._ordered_binary import binary


def _baseline_registers(coeffs, x):
    """Evaluate real binary64 arrays; caller handles empty/constant series."""
    if coeffs.ndim == 2:
        x = x[..., None]
    x2 = binary(jnp.asarray(2.0, x.dtype), x, 0)
    shape = jnp.broadcast_shapes(x.shape, coeffs.shape[1:])
    zero = jnp.zeros(shape, dtype=coeffs.dtype)
    degree = coeffs.shape[0] - 1

    def step(i, state):
        b1, b2 = state
        k = degree - 2 * i

        def instruction(registers, opcode):
            first, second, temporary = registers
            # A dynamic instruction selects exactly one source operation per
            # iteration. The loop carry separates multiply/add/subtract even
            # after compilation; an optimization barrier alone does not suffice.
            operations = (
                lambda: (first, second, x2 * first),
                lambda: (first, second, coeffs[k] + temporary),
                lambda: (first, temporary - second, temporary),
                lambda: (first, second, x2 * second),
                lambda: (first, second, coeffs[k - 1] + temporary),
                lambda: (temporary - first, second, temporary),
            )
            return jax.lax.switch(opcode, operations), None

        # The prior binary helper performed the operation followed by multiply
        # by one for finite operands, or the same operation for nonfinite ones.
        # The instruction boundary makes that identity step unnecessary here.
        # No finiteness assumption is made about inputs or intermediate values.
        (b1, b2, _), _ = jax.lax.scan(
            instruction,
            (b1, b2, zero),
            jnp.arange(6, dtype=jnp.int32),
            unroll=False,
        )
        return b1, b2

    b1, b2 = jax.lax.fori_loop(0, degree // 2, step, (zero, zero))
    if degree % 2:
        b1, b2 = binary(binary(coeffs[1], binary(x2, b1, 0), 1), b2, 2), b1
    half_x = binary(jnp.asarray(0.5, x.dtype), x2, 0)
    return binary(binary(coeffs[0], binary(half_x, b1, 0), 1), b2, 2)


def _same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    mask = ~np.isnan(b)
    np.testing.assert_array_equal(a[mask].view(np.uint64), b[mask].view(np.uint64))


@pytest.mark.parametrize("n", [1023, 1024, 1025, 2051])
@pytest.mark.parametrize("columns", [False, True])
def test_chunk_boundaries_and_columns(n, columns):
    c = (
        jnp.arange(1.0, 25.0).reshape(8, 3) / 25
        if columns
        else jnp.asarray([1.0, -0.0, 0.2, -0.3, 0.01, -0.005, 0.001])
    )
    x = jnp.linspace(-1.0, 1.0, n)
    _same_bits(_clenshaw(c, x), jax.jit(_baseline_registers)(c, x))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("position", [1023, 1024])
def test_global_nonfinite_choice_across_chunk(value, position):
    c = jnp.asarray([1.0, -0.0, 0.2, -0.3, 0.01, -0.005, 0.001, 0.002])
    x = jnp.linspace(-2.0, 2.0, 2051).at[position].set(value)
    _same_bits(_clenshaw(c, x), jax.jit(_baseline_registers)(c, x))


@pytest.mark.parametrize(
    "coeffs",
    [
        [0.0, -0.0, 0.0, -0.0],
        [1e308, 1e308, -1e308, 1e308],
        [1.0, np.inf, -np.inf, np.nan],
        [1e-310, -1e-310, 1e-310, -1e-310],
    ],
)
def test_ieee_chunk_seams(coeffs):
    x = jnp.tile(jnp.asarray([-2.0, -1.0, -0.0, 0.0, 1.0, 2.0]), 343)
    c = jnp.asarray(coeffs)
    _same_bits(_clenshaw(c, x), jax.jit(_baseline_registers)(c, x))


@pytest.mark.parametrize("shape", [(), (0,), (2, 1025)])
def test_scalar_empty_matrix_points(shape):
    x = jnp.zeros(shape)
    c = jnp.arange(24.0, dtype=jnp.float64).reshape(8, 3) / 25
    _same_bits(_clenshaw(c, x), jax.jit(_baseline_registers)(c, x))


@pytest.mark.parametrize("ncoeff", [0, 1])
def test_empty_constant_series_bypass(ncoeff):
    c = jnp.zeros((ncoeff, 3))
    x = jnp.zeros((1025,))
    np.testing.assert_array_equal(_clenshaw(c, x), np.zeros((1025, 3)))


@pytest.mark.parametrize("npoints", [1025, 2051])
def test_cross_chunk_joint_jvp_and_vjp(npoints):
    # Same100eps absolute analytic bound as the existing owned-JAX controls.
    # Mean-scaled cotangents keep the analytic reduction independent of npoints.
    c = np.arange(1.0, 19.0).reshape(6, 3) / 18
    dc = np.cos(np.arange(18.0)).reshape(6, 3) / 10
    x = np.linspace(-0.9, 0.9, npoints)
    dx = np.sin(np.arange(npoints)) / 10
    weight = np.arange(1.0, npoints * 3 + 1).reshape(npoints, 3) / (npoints * 3) ** 2
    basis = np.polynomial.chebyshev.chebvander(x, 5)
    derivative = np.polynomial.chebyshev.chebval(x, np.polynomial.chebyshev.chebder(c, axis=0)).T
    args = tuple(map(jnp.asarray, (c, x)))
    _, tangent = jax.jit(lambda a, b, da, db: jax.jvp(_clenshaw, (a, b), (da, db)))(
        *args, jnp.asarray(dc), jnp.asarray(dx)
    )
    bound = 100 * np.finfo(float).eps
    np.testing.assert_allclose(tangent, basis @ dc + dx[:, None] * derivative, rtol=0, atol=bound)
    _, pullback = jax.vjp(_clenshaw, *args)
    gc, gx = pullback(jnp.asarray(weight))
    np.testing.assert_allclose(gc, basis.T @ weight, rtol=0, atol=bound)
    np.testing.assert_allclose(gx, np.sum(weight * derivative, axis=1), rtol=0, atol=bound)
