"""Focused JAX controls for the source ratinterp pole/evaluation paths.

Provenance
----------
MATLAB source : ratinterp.m, tests/misc/test_ratinterp.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import _clenshaw
from chebfunjax.utils.ratapprox import (
    _assemble_matrices_rat,
    _chebtech1_vals2coeffs_matrix_apply,
    _chebyshev_roots,
    _construct_rat_approx,
    _eval_cheb_poly,
    ratinterp,
)


def test_complex_chebyshev_colleague_roots_jit():
    # Convert (x-r1)(x-r2) to c0+c1*T1+c2*T2. The nonconjugate roots
    # ensure realifying the colleague matrix's first column cannot pass.
    expected_roots = np.asarray([0.2 + 0.4j, -0.7 + 0.15j])
    r1, r2 = expected_roots
    coeffs = jnp.asarray([0.5 + r1 * r2, -(r1 + r2), 0.5])
    roots = jax.jit(_chebyshev_roots)(coeffs)
    np.testing.assert_allclose(
        np.sort_complex(np.asarray(roots)),
        np.sort_complex(expected_roots),
        atol=100 * np.finfo(float).eps,
        rtol=0,
    )
    values = _clenshaw(coeffs, roots)
    scale = np.max(np.abs(np.asarray(coeffs)))
    np.testing.assert_allclose(
        values, 0.0, atol=100 * np.finfo(float).eps * scale, rtol=0
    )


def test_complex_chebyshev_rational_closure_jit_and_shapes():
    numerator = jnp.asarray([1.0 + 2.0j, 0.5 - 0.25j])
    denominator = jnp.asarray([1.0 + 0.0j, -0.2 + 0.0j])
    r = _construct_rat_approx(
        "TYPE1", None, numerator, denominator, 1, 1, -1.0, 1.0
    )
    x = jnp.asarray([0.2 + 0.1j, -0.3 + 0.4j])
    result = jax.jit(r)(x)
    expected = (numerator[0] + numerator[1] * x) / (
        denominator[0] + denominator[1] * x
    )
    np.testing.assert_allclose(result, expected, atol=100 * np.finfo(float).eps, rtol=0)
    scalar_result = jax.jit(r)(jnp.asarray(0.2 + 0.1j))
    assert scalar_result.shape == ()
    assert jnp.iscomplexobj(scalar_result)

    x_real = 0.2
    derivative = jax.grad(lambda value: jnp.real(r(value)))(x_real)
    p = numerator[0] + numerator[1] * x_real
    q = denominator[0] + denominator[1] * x_real
    expected_derivative = jnp.real(
        (numerator[1] * q - p * denominator[1]) / q**2
    )
    np.testing.assert_allclose(
        derivative, expected_derivative, atol=100 * np.finfo(float).eps, rtol=0
    )

    empty_coeffs = jnp.empty((0,), dtype=jnp.complex128)
    empty = jax.jit(lambda value: _eval_cheb_poly(empty_coeffs, value))(x)
    np.testing.assert_array_equal(empty, jnp.zeros_like(x))
    constant = _construct_rat_approx(
        "TYPE1", None, jnp.asarray([1.0 + 2.0j]), jnp.asarray([2.0]), 0, 0,
        -1.0, 1.0,
    )
    constant_values = jax.jit(constant)(x)
    np.testing.assert_array_equal(constant_values, jnp.full(x.shape, 0.5 + 1.0j))


def test_complex_chebtech1_transform_jit_preserves_low_degree_modes():
    n = 5
    theta = (2 * jnp.arange(n - 1, -1, -1) + 1) * jnp.pi / (2 * n)
    x = jnp.cos(theta)
    scale = 1.0 + 0.75j
    vals = scale * (2.0 + 3.0 * x - 0.5 * (2.0 * x**2 - 1.0))
    matrix_apply = jax.jit(partial(_chebtech1_vals2coeffs_matrix_apply, N=n))
    coeffs = matrix_apply(vals[:, None])[:, 0]
    expected = scale * jnp.asarray([2.0, 3.0, -0.5, 0.0, 0.0])
    np.testing.assert_allclose(coeffs, expected, atol=100 * np.finfo(float).eps, rtol=0)


def test_complex_arbitrary_node_assembly_jit_uses_conjugate_projection():
    nodes = jnp.asarray([-0.8 + 0.1j, -0.1 + 0.5j, 0.4 - 0.3j, 0.9 + 0.2j])
    vals = jnp.asarray([1.0 + 0.5j, -0.2 + 1.1j, 0.8 - 0.6j, 2.0 + 0.1j])
    assemble = jax.jit(
        lambda f, xi: _assemble_matrices_rat(f, 1, xi, "ARBITRARY", 4)
    )
    z, r_qr, q_qr = assemble(vals, nodes)
    q_np = np.asarray(q_qr)
    expected = q_np.conj().T @ np.diag(np.asarray(vals)) @ q_np[:, :2]
    np.testing.assert_allclose(z, expected, atol=100 * np.finfo(float).eps, rtol=0)
    assert np.iscomplexobj(np.asarray(r_qr))


@pytest.mark.parametrize("grid", ["type0", "type1", "type2", "equi"])
def test_source_all_poles_retains_small_nonzero_imaginary_part(grid):
    pole = 0.2 + 1e-12j

    def f(x):
        return 1.0 / (x - pole)

    # TYPE0 with only two samples has empty symmetry comparisons; the
    # literal MATLAB check then marks both parities and drops the denominator.
    # Use eight samples to test the intended complex-pole contract. Keep the
    # original 100EPS accuracy bound unchanged.
    r, _, _, mu, nu, poles, _ = ratinterp(f, 0, 1, NN=8, xi=grid)
    assert (mu, nu) == (0, 1)
    pole = np.asarray(poles).reshape(-1)[0]
    bound = 100 * np.finfo(float).eps
    assert abs(pole.real - 0.2) <= bound
    assert abs(pole.imag - 1e-12) <= bound

    x = 0.35 + 0.25j
    expected = 1.0 / (x - (0.2 + 1e-12j))
    scale = max(1.0, abs(expected))
    np.testing.assert_allclose(r(x), expected, atol=100 * np.finfo(float).eps * scale, rtol=0)


@pytest.mark.parametrize("odd", [False, True])
def test_kind1_parity_numerator_modes_accept_jax_transform_output(odd):
    # ratinterp.m zeroes alternate numerator coefficients after reconstructing
    # the Chebtech1 numerator. These known even/odd rational functions exercise
    # that source branch with immutable transform output.
    def function(x):
        numerator = x if odd else 1.0
        return numerator / (1.0 + x**2)

    approximation, _, _, mu, nu, _, _ = ratinterp(
        function, 2, 2, NN=8, xi="type1"
    )
    assert (mu, nu) == (int(odd), 2)
    points = np.asarray([-0.7, 0.3, 0.6, 0.3 + 0.4j])
    np.testing.assert_allclose(
        approximation(points), function(points), atol=1e-10, rtol=0
    )
