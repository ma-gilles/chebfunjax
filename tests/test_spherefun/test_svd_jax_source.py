"""Independent surface measures and source SVD backend contracts.

Provenance
----------
MATLAB source : @spherefun/svd.m, tests/spherefun/test_svd.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Analytic controls below complement, and do not replace, the unchanged
three-clause source test. No source numerical fixture is used by the solver.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.tech.trigtech import Trigtech


def _tech(coeffs, real=True):
    return Trigtech.from_coeffs(jnp.asarray(coeffs), is_real=real)


def _field(cols, rows, pivots):
    return Spherefun(cols=cols, rows=rows, pivots=jnp.asarray(pivots),
                     idx_plus=tuple(range(len(cols))), idx_minus=())


@pytest.mark.parametrize("disable", [False, True])
@pytest.mark.parametrize("amplitude", [2.0, -3.0, 1.0 + 2.0j])
def test_rank_one_surface_measure(disable, amplitude, monkeypatch):
    # f = amplitude has squared surface norm 4*pi*|amplitude|^2.
    f = _field([_tech([amplitude], real=not isinstance(amplitude, complex))],
               [_tech([1.0])], [1.0])

    def forbidden(*args, **kwargs):
        raise AssertionError("NumPy linear algebra entered the JAX SVD route")

    monkeypatch.setattr(np.linalg, "qr", forbidden)
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    with jax.disable_jit(disable):
        s = f.svd()
    np.testing.assert_allclose(s, [abs(amplitude) * np.sqrt(4 * np.pi)],
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize("disable", [False, True])
def test_orthogonal_two_term_spectrum(disable):
    # f=1+cos(theta)*cos(lambda): independent factors are orthogonal.
    # Weighted latitude squared norms 2,2/3; longitude norms 2*pi,pi.
    one = _tech([1.0])
    cosine = _tech([0.5, 0.0, 0.5])
    f = _field([one, cosine], [one, cosine], [1.0, 1.0])
    with jax.disable_jit(disable):
        actual = f.svd()
    np.testing.assert_allclose(actual, np.sqrt([4 * np.pi, 2 * np.pi / 3]),
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize("disable", [False, True])
def test_empty_and_zero_cdr(disable):
    one = _tech([1.0])
    zero = _field([one], [one], [jnp.inf])
    with jax.disable_jit(disable):
        assert Spherefun.empty().svd().shape == (0,)
        np.testing.assert_array_equal(zero.svd(), [0.0])


def test_traced_rank_one_pivot():
    one = _tech([1.0])

    def value(pivot):
        return _field([one], [one], jnp.reshape(pivot, (1,))).svd()[0]

    pivot = jnp.asarray(2.0)
    np.testing.assert_allclose(jax.jit(value)(pivot), np.sqrt(np.pi),
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(jax.grad(value)(pivot), -np.sqrt(np.pi) / 2,
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize("disable", [False, True])
def test_source_nyquist_row_qr_branch_distinction(disable):
    from chebfunjax.spherefun._svd import _row_r

    # Stored even-length Nyquist cosine: source single-column innerProduct
    # resolves its square at 2*nf, while multiple-column QR uses n=nf.
    # This source aliasing distinction must not be silently oversampled away.
    nyquist = _tech([1.0, 0.0])
    one = _tech([1.0])
    with jax.disable_jit(disable):
        single = _row_r([nyquist])
        multiple = _row_r([nyquist, one])
    np.testing.assert_allclose(single, [[np.sqrt(np.pi)]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(jnp.abs(jnp.diag(multiple)),
                               np.sqrt([2 * np.pi, 2 * np.pi]),
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize("disable", [False, True])
def test_complex_pivot_and_zero_pivot_source_cdr(disable):
    one = _tech([1.0])
    with jax.disable_jit(disable):
        complex_s = _field([one], [one], [1.0 + 2.0j]).svd()
        zero_s = _field([one], [one], [0.0]).svd()
    np.testing.assert_allclose(complex_s, [np.sqrt(4 * np.pi / 5)],
                               rtol=0, atol=1e-12)
    np.testing.assert_array_equal(zero_s, [0.0])


@pytest.mark.parametrize("disable", [False, True])
def test_weighted_qr_asy_quadrature_branch(disable):
    # A padded constant probes the existing public legpts ASY dispatch
    # without creating a high-degree target solution fixture.
    coeffs = jnp.zeros((101,), dtype=jnp.complex128).at[50].set(1)
    col = Trigtech.from_coeffs(coeffs, is_real=True)
    f = _field([col], [_tech([1.0])], [1.0])
    with jax.disable_jit(disable):
        actual = f.svd()
    np.testing.assert_allclose(actual, [np.sqrt(4 * np.pi)], rtol=0, atol=1e-12)


@pytest.mark.parametrize("disable", [False, True])
def test_complex_two_term_nonconjugating_core(disable):
    # Independent finite matrix in orthonormal latitude/longitude bases.
    # Both complex rows and complex pivots distinguish .' from '.
    one = _tech([1.0])
    cosine = _tech([0.5, 0.0, 0.5])
    rows = [_tech([0.5j, 1.0, 0.5j], real=False),
            _tech([0.5, 2.0j, 0.5], real=False)]
    pivots = np.asarray([1 + 0.5j, 2 - 1j])
    f = _field([one, cosine], rows, pivots)
    row_coefficients = np.asarray([
        [np.sqrt(2 * np.pi), 1j * np.sqrt(np.pi)],
        [2j * np.sqrt(2 * np.pi), np.sqrt(np.pi)],
    ])
    expected_matrix = (np.diag(np.sqrt([2, 2 / 3]))
                       @ np.diag(1 / pivots) @ row_coefficients)
    expected = np.linalg.svd(expected_matrix, compute_uv=False)
    with jax.disable_jit(disable):
        actual = f.svd()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
