"""Source-contract checks for the large Legendre transform implementation.

Provenance: Chebfun commit 7574c77680d7e82b79626300bf255498271a72df,
``leg2cheb.m``, ``legpts.m``, ``@chebfun/idlt.m`` and
``tests/misc/test_leg2cheb.m``. MATLAB source tolerances are retained for
ported assertions. Additional ASY/JIT controls state their numerical bounds
at the assertion site; these are not claims of MATLAB test coverage.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import scipy.special

jax.config.update("jax_enable_x64", True)

from chebfunjax.utils.legendre_fast import _idlt_ndct_transpose_source
from chebfunjax.utils.quadrature import legpts
from chebfunjax.utils.transforms import (
    _legendre_idlt,
    leg2cheb,
    legcoeffs2legvals,
    ndct,
)

SOURCE_TOL = 1e-13


@pytest.mark.parametrize("n", [10, 5000])
def test_legendre_values_wrapper_complex_matrix_source_polynomials(n):
    # legcoeffs2legvals.m delegates to DLT, whose direct/18-term branches
    # both support matrix columns. This analytic oracle also catches the
    # former vector-shaped rolling recurrence for small inputs.
    nodes, _weights = legpts(n)
    coefficients = jnp.zeros((n, 2), dtype=jnp.complex128)
    coefficients = coefficients.at[0].set(jnp.array([1 + 2j, 0.5 - 0.25j]))
    coefficients = coefficients.at[1].set(jnp.array([-0.25 + 0.5j, 1.0 + 0.75j]))
    expected = coefficients[0][None, :] + nodes[:, None] * coefficients[1][None, :]
    actual = legcoeffs2legvals(coefficients)
    assert actual.shape == (n, 2)
    assert float(jnp.max(jnp.sum(jnp.abs(actual - expected), axis=1))) < float(
        100 * n * jnp.finfo(jnp.float64).eps
    )


@pytest.mark.parametrize("n", [0, 1])
def test_legendre_values_wrapper_empty_and_singleton_source_contract(n):
    coefficients = jnp.full((n, 2), 3.0, dtype=jnp.float64)
    expected = coefficients if n == 0 else jnp.ones_like(coefficients)
    actual = legcoeffs2legvals(coefficients)
    assert bool(jnp.array_equal(actual, expected))


@pytest.mark.parametrize("n", [512, 513])
def test_leg2cheb_pure_imaginary_low_modes(n):
    coefficients = jnp.zeros((n, 2), dtype=jnp.complex128)
    coefficients = coefficients.at[0, 0].set(2j)
    coefficients = coefficients.at[1, 1].set(0.5j)
    actual = leg2cheb(coefficients)
    assert float(jnp.max(jnp.abs(actual - coefficients))) < 10 * SOURCE_TOL
    if n <= 512:
        # Direct source idct1 delegates to Chebtech2's exact real/imag
        # handling. The source fast branch retains the complex FFT result
        # without this coercion; its roundoff is covered by the bound above.
        assert bool(jnp.all(jnp.real(actual) == 0))


def test_ndct_default_large_angles_match_analytic_high_mode():
    # Source one-argument NDCT requests four legpts outputs; reconstructing
    # acos(x) loses endpoint accuracy at this degree. The independent oracle
    # evaluates T_degree(cos(theta))=cos(degree*theta) at the source angles.
    n = 10006
    _nodes, _weights, _bary, theta = legpts(n, newtheta=True)
    coefficients = jnp.zeros(n, dtype=jnp.float64).at[-1].set(1.0)
    expected = jnp.cos((n - 1) * theta)
    actual = ndct(coefficients)
    # Additional high-degree control uses the source DLT test's 100*n*eps
    # allowance. This is not an original MATLAB NDCT test assertion.
    assert float(jnp.max(jnp.abs(actual - expected))) < float(
        100 * n * jnp.finfo(jnp.float64).eps
    )


def _matrix_fixture(n: int, ncols: int) -> jax.Array:
    index = jnp.arange(n * ncols, dtype=jnp.float64).reshape((n, ncols))
    return jnp.sin(0.13 * index) / (1.0 + index % 13)


@pytest.mark.parametrize("n", [512, 513])
def test_leg2cheb_threshold_batch_matches_independent_columns(n):
    """Source vectorization contract across the direct/fast branch boundary."""
    coefficients = _matrix_fixture(n, 3)
    batched = leg2cheb(coefficients)
    separate = jnp.stack(
        [leg2cheb(coefficients[:, col]) for col in range(coefficients.shape[1])],
        axis=1,
    )
    # MATLAB test_leg2cheb passes 7/8 use spectral matrix norm < tol=1e-13.
    assert float(jnp.linalg.norm(batched - separate, ord=2)) < SOURCE_TOL


@pytest.mark.parametrize("n", [512, 513])
def test_leg2cheb_threshold_transpose_matches_source_operator(n):
    """Compare both direct and fast transpose branches to the forward matrix."""
    matrix = leg2cheb(jnp.eye(n, dtype=jnp.float64))
    vector = jnp.sin(0.07 * jnp.arange(n, dtype=jnp.float64))
    expected = matrix.T @ vector
    actual = leg2cheb(vector, trans=True)
    # MATLAB test_leg2cheb pass 14 uses infinity norm < 10*tol.
    assert float(jnp.max(jnp.abs(actual - expected))) < 10 * SOURCE_TOL


@pytest.mark.parametrize("n", [512, 513])
def test_leg2cheb_normalize_matrix_low_modes(n):
    """Source 'normalize' scaling applies independently to every column."""
    coefficients = jnp.zeros((n, 2), dtype=jnp.float64)
    coefficients = coefficients.at[0, 0].set(1.0)
    coefficients = coefficients.at[1, 1].set(1.0)
    actual = leg2cheb(coefficients, normalize=True)
    expected = jnp.zeros_like(actual)
    expected = expected.at[0, 0].set(jnp.sqrt(0.5))
    expected = expected.at[1, 1].set(jnp.sqrt(1.5))
    # Source pass 10 checks the normalized constant norm within tol; pass 12
    # uses 10*tol for the large normalized inverse. This control verifies the
    # two-column source scaling/conversion at the branch threshold.
    assert float(jnp.max(jnp.abs(actual - expected))) < 10 * SOURCE_TOL


def test_leg2cheb_fast_workspace_exhaustion_raises_instead_of_returning_truncation():
    values = _matrix_fixture(513, 2)
    with pytest.raises(RuntimeError, match="did not reach source tolerance"):
        result = leg2cheb(values, max_rank=1)
        result.block_until_ready()


def test_leg2cheb_large_fast_branch_gradient_matches_transpose():
    n = 513
    direction = jnp.sin(0.09 * jnp.arange(n, dtype=jnp.float64))
    cotangent = jnp.cos(0.05 * jnp.arange(n, dtype=jnp.float64))

    def objective(coefficients):
        return jnp.vdot(cotangent, leg2cheb(coefficients))

    gradient = jax.grad(objective)(direction)
    transpose = leg2cheb(cotangent, trans=True)
    assert float(jnp.max(jnp.abs(gradient - transpose))) < 10 * SOURCE_TOL
    assert abs(
        float(jnp.vdot(cotangent, leg2cheb(direction)) - jnp.vdot(transpose, direction))
    ) < 10 * SOURCE_TOL


def _cosine_transpose(theta: jax.Array, weighted_values: jax.Array) -> jax.Array:
    frequency = jnp.arange(theta.shape[0], dtype=jnp.float64)
    matrix = jnp.cos(theta[:, None] * frequency[None, :])
    return matrix.T @ weighted_values


def test_idlt_source_pure_imaginary_input_returns_imaginary_coefficients():
    n = 32
    _nodes, weights, _bary, theta = legpts(n, newtheta=True)
    index = jnp.arange(n * 2, dtype=jnp.float64).reshape((n, 2))
    real_values = jnp.sin(0.21 * index) + 0.2 * jnp.cos(0.07 * index)
    weighted = 1j * weights[:, None] * real_values
    actual = _idlt_ndct_transpose_source(weighted, theta)
    expected = jnp.imag(_cosine_transpose(theta, weighted))
    assert not jnp.iscomplexobj(actual)
    # Independent source-equation control at moderate n; no source test gives
    # this private stage a standalone tolerance.
    assert float(jnp.max(jnp.abs(actual - expected))) < 1e-12


def test_idlt_large_branch_is_jittable_for_a_low_degree_polynomial():
    n = 5000
    nodes, _weights = legpts(n)
    actual = jax.jit(_legendre_idlt)(nodes)
    expected = jnp.zeros((n,), dtype=jnp.float64).at[1].set(1.0)
    # Analytic P1 oracle; allowance is 100*n*eps for the finite quadrature and
    # transform accumulation. MATLAB idlt.m selects its large branch at 5000.
    allowance = 100.0 * n * jnp.finfo(jnp.float64).eps
    assert actual.shape == (n,)
    assert float(jnp.max(jnp.abs(actual - expected))) < float(allowance)


@pytest.mark.parametrize("n", [100, 101, 251])
def test_legpts_asy_moderate_nodes_and_weights_against_scipy(n):
    nodes, weights, _bary, theta = legpts(n, newtheta=True)
    ref_nodes, ref_weights = scipy.special.roots_legendre(n)
    assert float(jnp.max(jnp.abs(nodes - jnp.asarray(ref_nodes)))) < 3e-14
    assert float(jnp.max(jnp.abs(weights - jnp.asarray(ref_weights)))) < 3e-14
    assert float(jnp.max(jnp.abs(nodes - jnp.cos(theta)))) < 2e-15


@pytest.mark.parametrize("n", [5000, 5001, 10000])
def test_legpts_asy_large_source_invariants(n):
    nodes, weights, bary, theta = legpts(n, newtheta=True)
    assert nodes.shape == weights.shape == bary.shape == theta.shape == (n,)
    assert bool(jnp.all(jnp.isfinite(nodes)))
    assert bool(jnp.all(jnp.isfinite(weights)))
    assert bool(jnp.all(jnp.isfinite(bary)))
    assert bool(jnp.all(jnp.isfinite(theta)))
    assert bool(jnp.all(jnp.diff(nodes) > 0.0))
    assert bool(jnp.all(weights > 0.0))
    assert bool(jnp.all(jnp.diff(theta) < 0.0))
    assert float(jnp.max(jnp.abs(nodes - jnp.cos(theta)))) < 2e-15
    assert abs(float(jnp.sum(weights)) - 2.0) < 2e-13
    assert abs(float(jnp.max(jnp.abs(bary))) - 1.0) < 2e-15
    expected_sign = jnp.where(jnp.arange(n) % 2 == 0, 1.0, -1.0)
    assert bool(jnp.array_equal(jnp.sign(bary), expected_sign))
