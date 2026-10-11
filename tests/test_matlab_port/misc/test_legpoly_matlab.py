"""Direct translations of the 26 unchanged predicates in tests/misc/test_legpoly.m."""
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import legpoly


def _legendre(degree, x):
    coefficients = np.zeros(degree + 1)
    coefficients[-1] = 1.0
    x = np.asarray(x)
    values = np.polynomial.legendre.legval(x, coefficients)
    # The exact endpoint identity P_n(±1)=(±1)^n avoids recurrence
    # roundoff in the test oracle at degree 1500; MATLAB legendre.m returns
    # these exact endpoint values in the retained native capture.
    values = np.where(x == 1.0, 1.0, values)
    values = np.where(x == -1.0, (-1.0)**degree, values)
    return values


def _gram(f):
    # MATLAB p'*p: conjugate transpose times the quasimatrix.
    return f.H @ f


def _eps():
    return jnp.finfo(jnp.float64).eps


def _vscale(f):
    return float(f.vscale)


def test_01_method1_legendre_evaluation():
    p = legpoly(range(900, 1101), [-1, 1], 0, 1)
    x = jnp.linspace(-1, 1, 10)
    err = jnp.linalg.norm(p[:, 0](x) - _legendre(900, x), ord=jnp.inf)
    assert err < 5e3 * _eps() * _vscale(p)


def test_02_method1_orthogonality():
    p = legpoly(range(900, 1101), [-1, 1], 0, 1)
    gram = _gram(p)
    err = jnp.linalg.norm(gram - jnp.diag(jnp.diag(gram)), ord=2)
    assert err < 5e1 * _eps() * _vscale(p)


def test_03_method1_orthonormality():
    p = legpoly(range(900, 1101), [-1, 1], 'normalize', 1)
    err = jnp.linalg.norm(_gram(p) - jnp.eye(201), ord=2)
    assert err < 5e3 * _eps() * _vscale(p)


def test_04_method1_legendre_evaluation_no_domain():
    p = legpoly(range(900, 1101), [-1, 1], 0, 1)
    x = jnp.linspace(-1, 1, 10)
    err = jnp.linalg.norm(p[:, 0](x) - _legendre(900, x), ord=jnp.inf)
    assert err < 5e3 * _eps() * _vscale(p)


def test_05_method1_orthogonality_no_domain():
    p = legpoly(range(900, 1101), [-1, 1], 0, 1)
    gram = _gram(p)
    err = jnp.linalg.norm(gram - jnp.diag(jnp.diag(gram)), ord=2)
    assert err < 5e1 * _eps() * _vscale(p)


def test_06_method1_orthonormality_no_domain():
    p = legpoly(range(900, 1101), 'normalize', 1)
    err = jnp.linalg.norm(_gram(p) - jnp.eye(201), ord=2)
    assert err < 5e3 * _eps() * _vscale(p)


def test_07_method1_orthogonality_mapped_domain():
    p = legpoly(range(900, 1101), [0, 10000], 'normalize', 1)
    gram = _gram(p)
    err = jnp.linalg.norm(gram - jnp.diag(jnp.diag(gram)), ord=2)
    assert err < 1e5 * _eps() * _vscale(p)


def test_08_method1_orthonormality_mapped_domain():
    p = legpoly(range(900, 1101), [0, 10000], 'normalize', 1)
    err = jnp.linalg.norm(_gram(p) - jnp.eye(201), ord=2)
    assert err < 1e5 * _eps() * _vscale(p)


def test_09_method2_legendre_evaluation_with_breaks():
    p = legpoly(40, [-1, .2, 1])
    x = jnp.linspace(-1, 1, 10)
    err = jnp.linalg.norm(p(x) - _legendre(40, x), ord=jnp.inf)
    assert err < 50 * _eps() * _vscale(p)


def test_10_method2_orthogonality_with_breaks():
    p = legpoly(range(1, 101), [-1, -.2, .3, 1])
    gram = _gram(p)
    # MATLAB norm(A) uses the matrix 2-norm, not the Frobenius norm.
    err = jnp.linalg.norm(gram - jnp.diag(jnp.diag(gram)), ord=2)
    assert err < 10 * _eps() * _vscale(p)


def test_11_method2_orthonormality_with_breaks():
    p = legpoly(range(1, 101), [-1, .145, 1], 'normalize')
    err = jnp.linalg.norm(_gram(p) - jnp.eye(100), ord=2)
    assert err < 100 * _eps() * _vscale(p)


def test_12_method2_legendre_evaluation_no_domain():
    p = legpoly(40)
    x = jnp.linspace(-1, 1, 10)
    err = jnp.linalg.norm(p(x) - _legendre(40, x), ord=jnp.inf)
    assert err < 50 * _eps() * _vscale(p)


def test_13_method2_orthogonality_no_domain():
    p = legpoly(range(1, 101))
    gram = _gram(p)
    err = jnp.linalg.norm(gram - jnp.diag(jnp.diag(gram)), ord=2)
    assert err < 10 * _eps() * _vscale(p)


def test_14_method2_orthonormality_no_domain():
    p = legpoly(range(1, 101), 'normalize')
    err = jnp.linalg.norm(_gram(p) - jnp.eye(100), ord=2)
    assert err < 100 * _eps() * _vscale(p)


def test_15_method2_orthogonality_mapped_domain():
    p = legpoly(range(1, 101), [0, 155, 3333, 10000])
    gram = _gram(p)
    err = jnp.linalg.norm(gram - jnp.diag(jnp.diag(gram)), ord=2)
    assert err < 1e6 * _eps() * _vscale(p)


def test_16_method2_orthonormality_mapped_domain():
    p = legpoly(range(1, 101), [0, 3333, 10000], 'normalize')
    err = jnp.linalg.norm(_gram(p) - jnp.eye(100), ord=2)
    assert err < 1e4 * _eps() * _vscale(p)


def test_17_method3_legendre_evaluation_with_breaks():
    p = legpoly(1500, [-1, -.5, -.3, 1])
    x = jnp.linspace(-1, 1, 10)
    err = jnp.linalg.norm(p(x) - _legendre(1500, x), ord=jnp.inf)
    assert err < 1e6 * _eps() * _vscale(p)


def test_18_method3_normalization_with_breaks():
    p = legpoly(1500, [-1, 0, 1], 'normalize')
    err = jnp.linalg.norm(_gram(p) - 1)
    assert err < 1e4 * _eps() * _vscale(p)


def test_19_method3_legendre_evaluation_no_domain():
    p = legpoly(1500)
    x = jnp.linspace(-1, 1, 10)
    err = jnp.linalg.norm(p(x) - _legendre(1500, x), ord=jnp.inf)
    assert err < 1e4 * _eps() * _vscale(p)


def test_20_method3_normalization_no_domain():
    p = legpoly(1500, 'normalize')
    err = jnp.linalg.norm(_gram(p) - 1)
    assert err < 1e4 * _eps() * _vscale(p)


def test_21_method3_vectorized_degree_1000():
    p = legpoly([1000, 1500])
    x = jnp.linspace(-1, 1, 10)
    err = jnp.linalg.norm(p[:, 0](x) - _legendre(1000, x), ord=jnp.inf)
    assert err < 1e4 * _eps() * _vscale(p)


def test_22_method3_vectorized_degree_1500():
    p = legpoly([1000, 1500])
    x = jnp.linspace(-1, 1, 10)
    err = jnp.linalg.norm(p[:, 1](x) - _legendre(1500, x), ord=jnp.inf)
    assert err < 1e4 * _eps() * _vscale(p)


def test_23_method3_vectorized_orthonormality():
    p = legpoly([1000, 1500], 'normalize')
    err = jnp.linalg.norm(_gram(p) - jnp.eye(2), ord=2)
    assert err < 1e4 * _eps() * _vscale(p)


def test_24_method2_row_input_orthogonality():
    p = legpoly(jnp.arange(1, 11)[:, None], [-1, .2, 1])
    assert p.is_transposed
    gram = p @ p.H
    err = jnp.linalg.norm(gram - jnp.diag(jnp.diag(gram)), ord=2)
    assert err < 5 * _eps() * _vscale(p)


def test_25_method2_row_input_orthonormality():
    p = legpoly(jnp.arange(1, 11)[:, None], [-1, .6, 1], 'normalize')
    assert p.is_transposed
    err = jnp.linalg.norm(p @ p.H - jnp.eye(10), ord=2)
    assert err < 10 * _eps() * _vscale(p)


def test_26_empty_normalized():
    p = legpoly([], [-1, .66, 1], 'normalize')
    assert p.isempty()


def test_method1_unsorted_duplicate_degree_columns():
    degrees = [3, 1, 3, 0, 1]
    p = legpoly(degrees, [-1, 1], 0, 1)
    x = jnp.linspace(-1, 1, 17)
    for column, degree in enumerate(degrees):
        err = jnp.max(jnp.abs(p[:, column](x) - _legendre(degree, x)))
        assert err < 50 * _eps() * _vscale(p)
