"""Literal scoped predicates, tests/chebfun/test_get.m at7574c77.

The existing mapped module remains supplemental. Source cell conventions are
explicit: level0 outer row of column-cells; level1 rectangular [piece][column];
ragged level1/2 outer row of column-cells. Quasimatrix is the existing public
structural adapter to source classarrays. Its columns come from actual public
column extraction, following cheb2quasi.m; no ONEFUN data are supplied.
This file retains literal original predicates31 and35-38, including their
native delta matrix norm and Gamma/exponent operands.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import get
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.domain import Domain


def _isnum(c):
    return hasattr(c, 'dtype') and c.dtype.kind in 'iufc'


def _matrix_cell(c):
    assert isinstance(c, list) and all(isinstance(row, list) for row in c)
    assert c and len({len(row) for row in c}) == 1
    return len(c), len(c[0])


def _column_cell(c):
    assert isinstance(c, list) and all(_isnum(leaf) for leaf in c)
    return len(c), 1


def _outer_cells(c):
    assert isinstance(c, list) and all(isinstance(v, list) for v in c)
    assert all(_isnum(leaf) for v in c for leaf in v)
    return 1, len(c)


@pytest.fixture(scope='module')
def source_objects():
    fs1 = cj.chebfun(jnp.sin, domain=(-1., 1.))
    fs2 = cj.chebfun(jnp.cos, domain=(-1., 0., 1.))
    def op(x):
        return jnp.stack((jnp.sin(x), jnp.cos(x)), axis=-1)
    fa1 = cj.chebfun(op, domain=(-1., 1.))
    fa2 = cj.chebfun(op, domain=(-1., 0., 1.))
    # Literal cheb2quasi column extraction through existing public objects.
    fq1 = Quasimatrix([fa1.extract_columns(j) for j in range(2)], fa1.domain)
    fq2 = Quasimatrix([fa2.extract_columns(j) for j in range(2)], fa2.domain)
    fq3 = Quasimatrix([fs1, fs2], Domain((-1., 0., 1.)))
    return fs1, fs2, fa1, fa2, fq1, fq2, fq3


@pytest.mark.parametrize('native_id', list(range(1, 32)))
def test_original_get_predicate(source_objects, native_id):
    fs1, fs2, fa1, fa2, fq1, fq2, fq3 = source_objects
    objects = source_objects
    if 1 <= native_id <= 7:
        c = get(objects[native_id-1], 'coeffs')
        if native_id in (1, 3, 5):
            assert _isnum(c) and c.shape[1] == (1 if native_id == 1 else 2)
        elif native_id in (2, 4, 6):
            assert _matrix_cell(c) == (2, 1 if native_id == 2 else 2)
        else:
            assert _outer_cells(c) == (1, 2)
            assert _column_cell(c[0]) == (1, 1)
            assert _column_cell(c[1]) == (2, 1)
    elif 8 <= native_id <= 14:
        c = get(objects[native_id-8], 'coeffs', 1)
        if native_id != 14:
            expected = [(1, 1), (2, 1), (1, 2), (2, 2), (1, 2), (2, 2)][native_id-8]
            assert _matrix_cell(c) == expected
        else:
            assert _outer_cells(c) == (1, 2)
            assert _column_cell(c[0]) == (1, 1)
            assert _column_cell(c[1]) == (2, 1)
    elif 15 <= native_id <= 21:
        c = get(objects[native_id-15], 'coeffs', 0)
        expected = [(1, 1), (1, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, None)][native_id-15]
        assert _outer_cells(c) == (1, expected[0])
        for j, column in enumerate(c):
            pieces = expected[1] if expected[1] is not None else (1 if j == 0 else 2)
            assert _column_cell(column) == (pieces, 1)
    elif 22 <= native_id <= 28:
        c = get(objects[native_id-22].T, 'coeffs')
        if native_id in (22, 24, 26):
            assert _isnum(c) and c.shape[0] == (1 if native_id == 22 else 2)
        elif native_id in (23, 25, 27):
            assert _matrix_cell(c) == ((1, 2) if native_id == 23 else (2, 2))
        else:
            # Source28 requires outer2x1, not the current ragged1x2 adapter.
            assert _matrix_cell(c) == (2, 1)
            assert _column_cell(c[0][0]) == (1, 1)
            assert _column_cell(c[1][0]) == (2, 1)
    elif native_id in (29, 30):
        x = cj.chebfun(lambda t: t)
        if native_id == 29:
            fd = fs1 + x.dirac()
            want = jnp.asarray([[0.], [1.]])
        else:
            fd = fs2 + (x-.5).dirac() + (x+.5).dirac()
            want = jnp.asarray([[-.5, .5], [1., 1.]])
        c = get(fd, 'deltas')
        assert c.shape == want.shape
        # Literal native matrix infinity norm (row sums), original5eps.
        assert float(jnp.max(jnp.sum(jnp.abs(c-want), axis=1))) < 5*jnp.finfo(jnp.float64).eps
    else:
        # Native source: fd3 = [fd1 fd2]; use the public quasimatrix adapter.
        x = cj.chebfun(lambda t: t)
        fd1 = fs1 + x.dirac()
        fd2 = fs2 + (x-.5).dirac() + (x+.5).dirac()
        fd3 = Quasimatrix([fd1, fd2], Domain((-1., 0., 1.)))
        c = get(fd3, 'deltas')
        wants = (jnp.asarray([[0.], [1.]]),
                 jnp.asarray([[-.5, .5], [1., 1.]]))
        assert isinstance(c, list) and len(c) == 2
        for actual, want in zip(c, wants):
            assert actual.shape == want.shape
            # Literal native matrix infinity norm (row sums), original5eps.
            assert float(jnp.max(jnp.sum(jnp.abs(actual-want), axis=1))) < 5*jnp.finfo(jnp.float64).eps


def test_quasimatrix_single_delta_column_collapses_like_native_get():
    x = cj.chebfun(lambda t: t)
    fd = cj.chebfun(jnp.sin, domain=(-1., 1.)) + x.dirac()
    one_column = Quasimatrix([fd], Domain((-1., 1.)))
    actual = get(one_column, 'deltas')
    want = jnp.asarray([[0.], [1.]])
    assert hasattr(actual, 'shape') and actual.shape == want.shape
    assert float(jnp.max(jnp.sum(jnp.abs(actual-want), axis=1))) < 5*jnp.finfo(jnp.float64).eps


@pytest.fixture(scope='module')
def fse1():
    return cj.chebfun(lambda t: 1/(t+2), domain=(-2., 2.), exps=(-1., 0.))


@pytest.mark.parametrize('native_id', [32, 33, 34])
def test_original_get_scalar_exponents(fse1, native_id):
    if native_id == 32:
        c = get(fse1, 'exponents')
        assert c.shape == (1, 2)
        np.testing.assert_array_equal(c, jnp.asarray([[-1., 0.]]))
    elif native_id == 33:
        assert _matrix_cell(get(fse1, 'exponents', 1)) == (1, 1)
    else:
        c = get(fse1, 'exponents', 0)
        assert _outer_cells(c) == (1, 1)
        assert _column_cell(c[0]) == (1, 1)


@pytest.fixture(scope='module')
def gamma_exponent_objects(fse1):
    from jax.scipy.special import gamma

    fse2 = cj.chebfun(gamma, domain=(-2., -1., 0., 2.),
                      exps=(-1., -1., -1., -1., -1., 0.))
    fqe1 = Quasimatrix([fse2, fse2], Domain((-2., 2.)))
    fqe2 = Quasimatrix([fse1, fse2], Domain((-2., 2.)))
    return fse2, fqe1, fqe2


@pytest.mark.parametrize('native_id', [35, 36, 37, 38])
def test_original_get_gamma_exponent_cells(gamma_exponent_objects, native_id):
    fse2, fqe1, fqe2 = gamma_exponent_objects
    want = (jnp.asarray([[-1., -1.]]),
            jnp.asarray([[-1., -1.]]),
            jnp.asarray([[-1., 0.]]))

    if native_id == 35:
        actual = get(fse2, 'exponents')
        assert actual.shape == (3, 2)
        np.testing.assert_array_equal(actual, jnp.concatenate(want, axis=0))
    elif native_id == 36:
        actual = get(fse2, 'exponents', 1)
        assert _matrix_cell(actual) == (3, 1)
        for row, expected in zip(actual, want):
            np.testing.assert_array_equal(row[0], expected)
    elif native_id == 37:
        actual = get(fqe1, 'exponents')
        assert _matrix_cell(actual) == (3, 2)
        for row, expected in zip(actual, want):
            np.testing.assert_array_equal(row[0], expected)
            np.testing.assert_array_equal(row[1], expected)
    else:
        actual = get(fqe2, 'exponents')
        assert isinstance(actual, list) and len(actual) == 2
        assert [len(column) for column in actual] == [1, 3]
        np.testing.assert_array_equal(actual[0][0], jnp.asarray([[-1., 0.]]))
        for leaf, expected in zip(actual[1], want):
            np.testing.assert_array_equal(leaf, expected)
