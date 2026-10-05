"""Independent controls for the source Laguerre GLR port.

Provenance: MATLAB ``lagpts.m`` GLR routines (Chebfun
7574c77680d7e82b79626300bf255498271a72df). SciPy roots are an independent
oracle used only in tests. The numerical references are independent of the production GLR recurrence.
"""
import numpy as np
import numpy.testing as npt
import pytest
from scipy.linalg import eigh_tridiagonal
from scipy.special import roots_genlaguerre

from chebfunjax.utils.quadrature import lagpts

EPS = np.finfo(np.float64).eps




def test_glr_initial_twenty_seeds_are_distinct_and_increasing_at_n42():
    roots = np.asarray(lagpts(42, method="GLR")[0])[:20]
    expected, _ = roots_genlaguerre(42, 0.0)
    npt.assert_allclose(roots, expected[:20], rtol=100 * EPS, atol=1000 * EPS)
    assert roots.shape == (20,)
    assert np.isfinite(roots).all()
    assert np.unique(roots).size == 20
    assert np.all(np.diff(roots) > 0.0)


@pytest.mark.parametrize('n', [5, 42, 128])
def test_explicit_glr_against_independent_alpha_zero_rule(n):
    x, w = map(np.asarray, lagpts(n, 0.0, method='GLR'))
    expected_x, expected_w = roots_genlaguerre(n, 0.0)
    npt.assert_allclose(x, expected_x, rtol=100 * EPS, atol=1000 * EPS)
    npt.assert_allclose(w, expected_w, rtol=100 * EPS, atol=1000 * EPS)
    assert np.all(np.diff(x) > 0) and np.all(w >= 0)
    assert abs(w.sum() - 1.0) <= 1000 * EPS


def test_explicit_glr_rejects_nonzero_generalized_parameter():
    with pytest.raises(ValueError, match='GLR method not supported for nonzero alpha'):
        lagpts(42, 0.5, method='GLR')


def test_default_n1000_alpha_zero_selects_glr_against_independent_tridiagonal():
    n = 1000
    x, w = map(np.asarray, lagpts(n, 0.0))
    explicit_x, explicit_w = map(np.asarray, lagpts(n, 0.0, method='GLR'))
    npt.assert_array_equal(x, explicit_x)
    npt.assert_array_equal(w, explicit_w)

    # Independent Golub--Welsch oracle, using scipy's symmetric tridiagonal
    # eigensolver rather than the production recurrence/GLR implementations.
    k = np.arange(1, n + 1, dtype=np.float64)
    off = np.arange(1, n, dtype=np.float64)
    expected_x, vectors = eigh_tridiagonal(2.0 * k - 1.0, off)
    expected_w = vectors[0, :] ** 2
    npt.assert_allclose(x, expected_x, rtol=100 * EPS, atol=1000 * EPS)
    npt.assert_allclose(w, expected_w, rtol=100 * EPS, atol=1000 * EPS)
    assert np.isfinite(x).all() and np.isfinite(w).all()
    assert np.all(np.diff(x) > 0) and np.all(w >= 0)
    # Moment bounds use the large-n multiplier already present in source
    # test_lagpts.m's n=251 pass, not a fabricated n=1000 MATLAB assertion.
    assert abs(w @ x - 1.0) <= 100_000 * EPS
    assert abs(w @ (x * x) - 2.0) <= 100_000 * EPS
