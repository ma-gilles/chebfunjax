"""Public LU controls supplementing the eight unchanged native predicates.

Provenance
----------
MATLAB source : @chebfun/lu.m, @chebfun/minandmax.m
Chebfun commit: 7574c77
"""

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.chebfun1d.linalg import Quasimatrix, lu


def test_empty_and_transpose():
    for empty in (Chebfun.empty(), [], ()):
        L, U, p = lu(empty)
        assert isinstance(L, Chebfun) and L.isempty()
        assert isinstance(U, jax.Array) and U.shape == (0, 0)
        assert isinstance(p, jax.Array) and p.shape == (0,)
    x = chebfun(lambda t: t)
    for row in (x.T, Quasimatrix([x.T], x.domain), [x.T]):
        with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:lu:sizes'):
            lu(row)


@pytest.mark.parametrize('adapter', ['list', 'quasi', 'array'])
@pytest.mark.parametrize('complex_values', [False, True])
def test_public_adapters_preserve_values(adapter, complex_values):
    x = chebfun(lambda t: t)
    cols = ([1 + 1j*x, x + 2j, x**2 - 1j*x] if complex_values
            else [0*x + 1, x, x**2])
    if adapter == 'list':
        A = cols
    elif adapter == 'quasi':
        A = Quasimatrix(cols, x.domain)
    else:
        A = chebfun(lambda t: jnp.stack(
            [1 + 1j*t, t + 2j, t**2 - 1j*t] if complex_values else
            [jnp.ones_like(t), t, t**2], axis=-1))
    L, U, p = lu(A)
    assert isinstance(U, jax.Array) and isinstance(p, jax.Array)
    assert bool(jnp.iscomplexobj(U)) == complex_values
    q = jnp.linspace(-1, 1, 33)
    expected = jnp.stack([c(q) for c in cols], axis=-1)
    assert float(jnp.max(jnp.abs(L(q) @ U - expected))) < 1e-13
    assert float(jnp.max(jnp.abs(jnp.diag(L(p)) - 1))) < 2e-14
    assert float(jnp.max(jnp.abs(jnp.triu(L(p), 1)))) < 2e-14


def test_rank_deficient_source_error():
    x = chebfun(lambda t: t)
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:rdivide:columnRdivide:divisionByZero'):
        lu([x, x])


def test_scalar_chebfun():
    x = chebfun(lambda t: 2 + t)
    L, U, p = x.lu()
    assert U.shape == (1, 1) and p.shape == (1,)
    assert float(p[0]) == 1.0
    assert float(jnp.abs(U[0, 0] - 3)) < 2e-14
    assert float(jnp.max(jnp.abs(L(jnp.array([-1., 0., 1.]))[:, 0]
                                     * U[0, 0] - jnp.array([1., 2., 3.])))) < 2e-14


def test_quasimatrix_method_and_constant_midpoint():
    x = chebfun(lambda t: t)
    A = Quasimatrix([0*x + 1, x], x.domain)
    L, U, p = A.lu()
    assert float(p[0]) == 0.0
    assert float(jnp.max(jnp.abs(L(p) @ U - A(p)))) < 2e-14


def test_duplicate_constant_pivot_error():
    one = chebfun(lambda t: jnp.ones_like(t))
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:lu:pivot'):
        lu([one, one])


def test_complex_extrema_keep_modulus_locations():
    f = chebfun(lambda t: 1 + 1j*t)
    (xmin, vmin), (xmax, vmax) = f.minandmax()
    assert abs(float(xmin)) < 2e-14
    assert abs(abs(float(xmax)) - 1) < 2e-14
    assert abs(complex(vmin) - 1) < 2e-14
    assert abs(abs(complex(vmax)) - 2**0.5) < 2e-14
