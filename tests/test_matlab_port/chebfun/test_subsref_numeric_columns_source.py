"""Native subsref pass12 and Python zero-based numeric-column adapters.

Provenance
----------
MATLAB source: tests/chebfun/test_subsref.m, @chebfun/subsref.m.
Chebfun commit: 7574c77.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from scipy.io import loadmat

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain


def _f():
    return Chebfun.from_coeffs(jnp.asarray([[1., 2., 3.], [.5, -1., 2.]]))


def test_native_pass12_exact_selection():
    xr = jnp.asarray(loadmat(Path(__file__).with_name("data")
                            / "feval_seedRNG_7681_xr.mat")["xr"])
    f = cj.chebfun(lambda x: jnp.stack(
        [jnp.sin(x-.1), jnp.cos(x-.2)], axis=-1))
    # Literal MATLAB Nx1 query: selection retains the matrix column.
    assert jnp.array_equal(f(xr, 1), cj.feval(f, xr)[:, 1:2])
    # Existing Python vector query convention drops an integer component axis.
    x = xr.ravel()
    assert jnp.array_equal(f(x, 1), cj.feval(f, x)[:, 1])
    assert jnp.array_equal(f(x, columns=1), cj.feval(f, x)[:, 1])


@pytest.mark.parametrize("shape", [(), (6,), (2, 3), (2, 3, 2), (2, 1, 2, 3)])
@pytest.mark.parametrize("row", [False, True])
def test_packed_order_duplicates_and_orientation(shape, row):
    f = _f()
    x = jnp.zeros(shape)
    full = f(x)
    if len(shape) < 2:
        expected = full[..., jnp.asarray([2, 0, 2])]
    else:
        w = shape[1]
        blocks = jnp.concatenate([jnp.arange(w)+2*w, jnp.arange(w),
                                  jnp.arange(w)+2*w])
        expected = jnp.take(full, blocks, axis=1)
    if row:
        f = f.T
        if expected.ndim >= 2:
            expected = jnp.swapaxes(expected, 0, 1)
    assert jnp.array_equal(f(x, [-1, 0, 2]), expected)
    assert jnp.array_equal(f(x, columns=[2, 0, 2]), expected)


@pytest.mark.parametrize("row", [False, True])
def test_slices_empty_scalar_and_jit(row):
    f = _f().T if row else _f()
    x = jnp.asarray([-.7, .2, .8])
    expected = _f()(x)[:, ::-1]
    if row:
        expected = expected.T
    assert jnp.array_equal(f(x, slice(None, None, -1)), expected)
    assert f(x, []).shape == ((0, 3) if row else (3, 0))
    assert f(0., []).shape == (0,)
    assert f(jnp.zeros((2, 3, 4)), []).shape == (
        (0, 2, 4) if row else (2, 0, 4))
    assert jnp.array_equal(jax.jit(lambda q: f(q, columns=[2, 0]))(x),
                           jax.jit(lambda q: _f()(q))(x)[:, [2, 0]].T
                           if row else jax.jit(lambda q: _f()(q))(x)[:, [2, 0]])
    scalar = Chebfun.from_coeffs(jnp.asarray([1., 2.]))
    assert jnp.array_equal(scalar(x, 0), scalar(x))
    assert scalar(x, [0]).shape == (3, 1)


def test_pointvalues_side_and_feval_distinction():
    f = Chebfun(funs=[_Piece.from_coeffs(jnp.asarray([[1., 2.]]), -1., 0.),
                     _Piece.from_coeffs(jnp.asarray([[3., 4.]]), 0., 1.)],
                domain=Domain((-1., 0., 1.))).set_point_values(
                    jnp.asarray([[1., 2.], [7., 8.], [3., 4.]]))
    assert f(0., 1) == 8.
    assert f(0., columns=1, side="left") == 2.
    assert jnp.array_equal(f(0., "right"), f(0., side="right"))
    for bad in [1, []]:
        with pytest.raises(ValueError) as caught:
            f(0., side=bad)
        assert caught.value.identifier == "CHEBFUN:CHEBFUN:feval:leftRight"
        with pytest.raises(ValueError) as caught:
            cj.feval(f, 0., bad)
        assert caught.value.identifier == "CHEBFUN:CHEBFUN:feval:leftRight"
    assert cj.feval(lambda x, a: x+a, 2, 3) == 5
    assert cj.feval(cj.chebfun(), 0., 1).size == 0
    with pytest.raises(TypeError):
        f(0., 0, columns=1)
    with pytest.raises(TypeError):
        f(0., "left", side="right")


@pytest.mark.parametrize("bad", [3, -4, 1.5, [1.5], True, [[0, 1]],
                                 jnp.asarray(True), jnp.asarray([True])])
def test_invalid_indices(bad):
    with pytest.raises((IndexError, TypeError, ValueError)):
        _f()(0., columns=bad)


def test_complex_values_and_empty_queries():
    f = Chebfun.from_coeffs(jnp.asarray([[1.+2j, 3.-1j], [.5j, 2.]]))
    x = jnp.asarray([[.2+.1j, -.4j], [.1, -.2]])
    full = f(x)
    assert jnp.array_equal(f(x, [1, 0]), full[:, jnp.asarray([2, 3, 0, 1])])
    assert f(x, 0).dtype == jnp.complex128
    assert f(jnp.zeros((0,)), []).shape == (0, 0)
    assert f(jnp.zeros((0, 3)), []).shape == (0, 0)
    assert f.T(jnp.zeros((2, 0)), []).shape == (0, 2)
