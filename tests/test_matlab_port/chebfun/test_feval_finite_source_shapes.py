"""Finite input-shape predicates from MATLAB chebfun/test_feval.m.

The query vector is captured from native seedRNG(7681); this test preserves
MATLAB column-major reshape and second-dimension component packing.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.io import loadmat

import chebfunjax as cj

_EPS = np.finfo(np.float64).eps
_XR = loadmat(Path(__file__).with_name("data") / "feval_seedRNG_7681_xr.mat")["xr"]


def _source_grid(shape):
    return np.reshape(_XR, shape, order="F")


def _array_values(x):
    """MATLAB `[sin(x), abs(x), exp(1i*x)]` component-block output."""
    x = np.asarray(x)
    return np.concatenate((np.sin(x), np.abs(x), np.exp(1j * x)), axis=1)


def _scalar_fun():
    return cj.chebfun(lambda x: jnp.cos(x - 0.2), domain=(-1.0, 1.0))


def _three_column_fun():
    exact = lambda x: jnp.stack((jnp.sin(x), jnp.abs(x), jnp.exp(1j*x)), axis=-1)
    return cj.chebfun(exact, domain=(-1.0, 1.0), splitting=True)


class TestFevalNativeFiniteShapes:
    def test_pass_1_literal_scalar_empty_matrix(self):
        f = cj.chebfun(lambda x: x)
        assert np.asarray(f(jnp.zeros((0, 4)))).shape == (0, 4)

    def test_pass_1_empty_matrix_shape_before_component_packing(self):
        exact = lambda x: jnp.stack((x, x**2, jnp.sin(x)), axis=-1)
        f = cj.chebfun(exact).transpose()
        # Native feval returns zeros(size(x)) before row transpose or output
        # packing, even for a transposed array-valued Chebfun.
        got = np.asarray(f(jnp.zeros((0, 4))))
        assert got.shape == (0, 4)

    @pytest.mark.parametrize(
        ("x", "expected_shape"),
        [
            pytest.param(_XR.T, (1, 1000), id="pass14-row"),
            pytest.param(_source_grid((100, 10)), (100, 10), id="pass15-matrix"),
            pytest.param(_source_grid((5, 20, 10)), (5, 20, 10), id="pass16-rank3"),
            pytest.param(_source_grid((5, 4, 5, 10)), (5, 4, 5, 10), id="pass17-rank4"),
        ],
    )
    def test_passes_14_to_17_scalar_shape_and_values(self, x, expected_shape):
        f = _scalar_fun()
        got = np.asarray(f(jnp.asarray(x)))
        assert got.shape == expected_shape
        assert np.max(np.abs(got - np.cos(x - 0.2))) < 10 * _EPS * float(f.vscale)

    def test_pass_18_transposed_scalar_matrix(self):
        f = _scalar_fun().transpose()
        x = _source_grid((100, 10))
        got = np.asarray(f(jnp.asarray(x)))
        assert got.shape == (10, 100)
        assert np.max(np.abs(got - np.cos(x - 0.2).T)) < 10 * _EPS * float(f.vscale)

    def test_pass_19_transposed_scalar_rank3(self):
        f = _scalar_fun().transpose()
        x = _source_grid((5, 20, 10))
        got = np.asarray(f(jnp.asarray(x)))
        assert got.shape == (20, 5, 10)
        expected = np.transpose(np.cos(x - 0.2), (1, 0, 2))
        assert np.max(np.abs(got - expected)) < 10 * _EPS * float(f.vscale)

    def test_pass_20_complex_and_outside_points(self):
        f = _scalar_fun()
        x = np.asarray([[-1 - 1e-6, 1e-6 + 1e-6j, 1 + 1e-6]])
        got = np.asarray(f(jnp.asarray(x)))
        assert np.max(np.abs(got - np.cos(x - 0.2))) < 10 * _EPS * float(f.vscale)

    def test_pass_21_array_vector(self):
        f = _three_column_fun()
        got = np.asarray(f(jnp.asarray(_XR)))
        expected = _array_values(_XR)
        assert got.shape == expected.shape
        assert np.max(np.abs(got - expected), axis=0).max() < 100 * _EPS * float(f.vscale)

    @pytest.mark.parametrize(
        ("x", "expected_shape"),
        [
            pytest.param(_source_grid((100, 10)), (100, 30), id="pass22-matrix"),
            pytest.param(_source_grid((5, 20, 10)), (5, 60, 10), id="pass23-rank3"),
            pytest.param(_source_grid((5, 4, 5, 10)), (5, 12, 5, 10), id="pass24-rank4"),
        ],
    )
    def test_passes_22_to_24_array_shape_and_values(self, x, expected_shape):
        f = _three_column_fun()
        got = np.asarray(f(jnp.asarray(x)))
        assert got.shape == expected_shape
        assert np.max(np.abs(got - _array_values(x))) < 100 * _EPS * float(f.vscale)

    def test_pass_25_transposed_array_rank4(self):
        f = _three_column_fun().transpose()
        x = _source_grid((5, 4, 5, 10))
        got = np.asarray(f(jnp.asarray(x)))
        assert got.shape == (12, 5, 5, 10)
        expected = np.swapaxes(_array_values(x), 0, 1)
        assert np.max(np.abs(got - expected)) < 100 * _EPS * float(f.vscale)

    @pytest.mark.parametrize(
        ("endpoint", "expected"),
        [
            pytest.param("left", [np.sin(-1), 1, np.exp(-1j)], id="pass26-left"),
            pytest.param("right", [np.sin(1), 1, np.exp(1j)], id="pass27-right"),
        ],
    )
    def test_passes_26_and_27_transposed_array_endpoint_values(self, endpoint, expected):
        # Native uses the same split array-valued F=[sin(x),abs(x),exp(i*x)]
        # as passes 21-25, sets isTransposed=1, then evaluates the endpoint.
        exact = lambda x: jnp.stack((jnp.sin(x), jnp.abs(x), jnp.exp(1j*x)), axis=-1)
        f = cj.chebfun(exact, domain=(-1.0, 1.0), splitting=True).transpose()
        got = np.asarray(f(endpoint))
        # Endpoint aliases return stored pointValues directly; unlike a
        # numeric scalar call, the transposed array-valued result is a column.
        assert got.shape == (3, 1)
        assert np.max(np.abs(got[:, 0] - expected)) < 10 * _EPS * float(f.vscale)
        numeric_scalar = np.asarray(f(-1.0 if endpoint == "left" else 1.0))
        assert numeric_scalar.shape == (3,)  # explicit Python scalar adapter

    @pytest.mark.parametrize(
        ("side", "expected"),
        [pytest.param("left", [1, 2], id="pass28-left"),
         pytest.param("right", [3, 4], id="pass29-right")],
    )
    def test_passes_28_and_29_piecewise_array_one_sided_values(self, side, expected):
        f = cj.chebfun(
            [lambda x: jnp.stack((x, 2*x), axis=-1),
             lambda x: jnp.stack((3*x, 4*x), axis=-1)],
            domain=(0.0, 1.0, 2.0),
        )
        got = np.asarray(f(1.0, side=side))
        # Native norm(row_error, inf) is the matrix infinity norm: with one
        # row, it sums both absolute component errors.
        assert np.sum(np.abs(got - expected)) < 10 * _EPS * float(f.vscale)

    def test_pass_30_array_evaluation_at_domain_breakpoints(self):
        exact = lambda x: jnp.stack((jnp.sin(x), jnp.cos(x)), axis=-1)
        f = cj.chebfun(exact, domain=(-1.0, 1.0))
        x = np.asarray([[-1.0], [0.5], [1.0]])
        got = np.asarray(f(jnp.asarray(x)))
        expected = np.concatenate((np.sin(x), np.cos(x)), axis=1)
        assert got.shape == expected.shape
        assert np.max(np.abs(got - expected)) < 10 * _EPS * float(f.vscale)


@pytest.mark.parametrize(
    ("call", "identifier"),
    [
        pytest.param(lambda f: f("???"),
                     "CHEBFUN:CHEBFUN:feval:strInput", id="endpoint-string"),
        pytest.param(lambda f: f(0.0, side="???"),
                     "CHEBFUN:CHEBFUN:feval:leftRightChar", id="side-string"),
        pytest.param(lambda f: f(0.0, side=1),
                     "CHEBFUN:CHEBFUN:feval:leftRight", id="side-nonstring"),
    ],
)
def test_native_feval_error_identifiers(call, identifier):
    f = _scalar_fun()
    with pytest.raises(ValueError) as exc_info:
        call(f)
    assert exc_info.value.identifier == identifier


def test_one_sided_array_matrix_packs_components_by_second_dimension():
    f = cj.chebfun(
        [lambda x: jnp.stack((x, 2*x), axis=-1),
         lambda x: jnp.stack((3*x, 4*x), axis=-1)],
        domain=(0.0, 1.0, 2.0),
    )
    x = np.asarray([[0.25, 1.0], [1.5, 1.75]])
    got = np.asarray(f(jnp.asarray(x), side="right"))
    expected = np.concatenate((np.where(x < 1.0, x, 3*x),
                               np.where(x < 1.0, 2*x, 4*x)), axis=1)
    assert got.shape == (2, 4)
    assert np.max(np.abs(got - expected)) < 10 * _EPS * float(f.vscale)


def test_matrix_point_values_and_row_orientation_eager_and_jit():
    f = cj.chebfun(lambda x: jnp.stack((x, 1+x), axis=-1))
    f = f.set_point_values(jnp.asarray([[10., -10.], [20., -20.]]))
    x = jnp.asarray([[-1., .25, 1.], [.5, 1., -1.]])
    expected = np.concatenate(([[10., .25, 20.], [.5, 20., 10.]],
                               [[-10., 1.25, -20.], [1.5, -20., -10.]]), axis=1)
    for fun, exact in ((f, expected), (f.transpose(), expected.T)):
        for evaluate in (fun, jax.jit(lambda z: fun(z))):
            actual = np.asarray(evaluate(x))
            assert actual.shape == exact.shape
            assert np.max(np.abs(actual-exact)) < 10 * _EPS * 20
    empty = jnp.empty((0, 4))
    assert jax.jit(lambda z: f.transpose()(z))(empty).shape == (0, 4)
