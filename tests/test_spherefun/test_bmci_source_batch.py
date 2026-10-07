"""Independent coefficient constraints for source BMC-I group projection.

Provenance
----------
MATLAB source : @spherefun/projectOntoBMCI.m; @trigtech/real.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun._bmci import _columns, _real_source, _rows
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize('disabled', [False, True])
def test_orthogonal_even_projection_independent_constraints(disabled):
    n = 7
    x = np.arange(1., n + 1)
    # Equality constraints: mirror symmetry and zero values at both poles.
    eye = np.eye(n)
    a = np.concatenate((eye[:n//2] - eye[::-1][:n//2],
                        np.ones((1, n)), ((-1.)**np.arange(n))[None]), axis=0)
    expected = x - np.linalg.pinv(a) @ (a @ x)
    with jax.disable_jit(disabled):
        got = jax.jit(lambda c: _columns(c, True, False))(jnp.asarray(x[:, None]))
    np.testing.assert_allclose(np.asarray(got)[:, 0], expected, rtol=0, atol=3e-14)


def test_short_column_uses_common_factor_space():
    constant = Trigtech(coeffs=jnp.asarray([1.]), is_real=True)
    high = Trigtech(coeffs=jnp.asarray([.5, 0., 0., 0., .5]), is_real=True)
    f = Spherefun(cols=[constant, high], rows=[constant, constant],
                  pivots=jnp.ones(2), idx_plus=(0, 1), idx_minus=(), nonzero_poles=False)
    out = f.projectOntoBMCI()
    np.testing.assert_allclose(out.cols[0].coeffs, [-1/3, 0, 2/3, 0, -1/3],
                               rtol=0, atol=5e-16)


def test_common_length_is_retained_across_group_extraction():
    constant = Trigtech(coeffs=jnp.asarray([1.]), is_real=True)
    high = Trigtech(coeffs=jnp.asarray([.5, 0., 0., 0., .5]), is_real=True)
    f = Spherefun(cols=[constant, high], rows=[constant, constant],
                  pivots=jnp.ones(2), idx_plus=(0,), idx_minus=(1,), nonzero_poles=False)
    out = f.projectOntoBMCI()
    assert out.cols[0].coeffs.shape == (5,)
    np.testing.assert_allclose(out.cols[0].coeffs, [-1/3, 0, 2/3, 0, -1/3],
                               rtol=0, atol=5e-16)


def test_even_nyquist_restored_before_real_early_return():
    c = jnp.asarray([[2.], [0.], [0.], [0.]])
    projected = _columns(c, True, True)
    assert projected.shape == (4, 1)
    np.testing.assert_array_equal(projected, c)
    np.testing.assert_array_equal(_real_source(projected), c)


@pytest.mark.parametrize('n', [4, 5])
@pytest.mark.parametrize('even', [False, True])
def test_row_mask_uses_original_storage_modes(n, even):
    c = jnp.ones((n, 2))
    got = _rows(c, even)
    modes = np.arange(n) - n//2
    expected = np.repeat(((modes % 2 == 0) if even else (modes % 2 != 0))[:, None], 2, axis=1)
    np.testing.assert_array_equal(got, expected)


def test_real_conversion_pure_imaginary_is_zero():
    got = _real_source(jnp.asarray([[2j]]))
    np.testing.assert_array_equal(got, [[0.]])


def test_real_conversion_simplifies_group_to_common_cutoff():
    coeffs = jnp.zeros((17, 2), dtype=jnp.complex128).at[8].set(jnp.asarray([2., 3j]))
    got = _real_source(coeffs)
    assert got.shape == (1, 2)
    np.testing.assert_allclose(got, [[2., 0.]], rtol=0, atol=2e-15)
