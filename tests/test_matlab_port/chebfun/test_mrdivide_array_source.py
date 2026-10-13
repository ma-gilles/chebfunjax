"""Literal array-valued clauses 1--5 from native test_mrdivide.m.

Provenance
----------
MATLAB source : tests/chebfun/test_mrdivide.m, clauses 1--5
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import Chebfun, Domain, chebpoly, legpoly, mrdivide

EPS = np.finfo(np.float64).eps


def test_native_mrdivide_array_valued_clauses_1_to_5():
    # Keep the source construction order and array-valued representations.
    T = chebpoly(range(4)).restrict([-1, -0.5, 0, 0.5, 1])
    L = legpoly(range(4)).restrict([-1, 0, 1])

    # pass(1): normest(T/2 - .5*T) < 10*eps.
    residual = mrdivide(T, 2) - 0.5 * T
    assert float(residual.normest()) < 10 * EPS

    # pass(2): (1:4)/T, evaluated at zero.
    x = mrdivide(jnp.arange(1, 5), T)
    assert abs(complex(x(0)) - (-2.625)) < 100 * EPS

    # pass(3): eye(4)/L, retaining the source's four output components.
    X = mrdivide(jnp.eye(4), L)
    x0 = np.asarray(X(0)).reshape(-1)
    assert np.linalg.norm(x0 - np.array([0.5, 0, -1.25, 0])) < 1e2 * EPS

    # pass(4): T/(1:4), evaluated at zero.
    x = mrdivide(T, jnp.arange(1, 5))
    assert abs(complex(x(0)) - (-1 / 15)) < 10 * EPS

    # pass(5): row/row array-valued solve and native infinity norm.
    X = mrdivide(T.T, L.T)
    C = np.diag([1, 1, 4 / 3, 8 / 5])
    C[2, 0] = -1 / 3
    C[3, 1] = -3 / 5
    assert np.linalg.norm(np.asarray(X) - C, ord=np.inf) < 1e2 * EPS


def test_mrdivide_complex_numeric_matrix_dual_control():
    """Check conjugation and output orientation in the numeric-matrix dual."""
    f = Chebfun.from_function(
        lambda x: jnp.stack((1 + 1j*x, 2 - 0.5j*x), axis=-1),
        domain=Domain((-1, 1)))
    matrix = jnp.asarray([[1, 1j], [1j, 1]])
    result = mrdivide(f, matrix)
    query = jnp.asarray([-0.75, -0.1, 0.4, 0.9])
    expected = f(query) @ jnp.linalg.inv(matrix)
    assert not result.is_transposed
    assert result.n_cols == 2
    assert jnp.max(jnp.abs(result(query) - expected)) < 100 * EPS
