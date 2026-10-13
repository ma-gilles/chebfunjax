"""IEEE interval routing from @chebfun/feval.m, native pin 7574c77.

Source-derived expectations use constant pieces to isolate routing from the
accuracy of floating-point polynomial evaluation near subnormal intervals.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain


def _constant_pieces(middle):
    breaks = (-1.0, middle, 0.5, 1.0)
    pieces = [_Piece.from_coeffs(jnp.asarray([float(i + 1)]), a, b)
              for i, (a, b) in enumerate(zip(breaks[:-1], breaks[1:]))]
    return Chebfun(funs=pieces, domain=Domain(breaks)).set_point_values(
        jnp.asarray([-3.0, 7.0, 8.0, 4.0]))


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize('negative', [False, True])
def test_public_subnormal_routing_eager_and_jit(dtype, negative):
    narrow = dtype in (np.float32, np.complex64)
    uint, real, center, sign = ((np.uint32, np.float32, 0x00400000, 0x80000000)
                                if narrow else
                                (np.uint64, np.float64, 0x0008000000000000, 0x8000000000000000))
    words = [center - 1, center, center + 1]
    if negative:
        words = [(v | sign) for v in words[::-1]]
    query = np.asarray(words, dtype=uint).view(real)
    middle = float(query[1])
    if np.issubdtype(dtype, np.complexfloating):
        z = np.zeros(query.shape, dtype=dtype)
        z.real = query
        query = z
    x = jnp.asarray(query)
    f = _constant_pieces(middle)
    expected = np.asarray([1.0, 7.0, 2.0])
    np.testing.assert_array_equal(np.asarray(f(x)), expected)
    np.testing.assert_array_equal(np.asarray(jax.jit(lambda z: f(z))(x)), expected)
    # Same comparison after fusion with arithmetic using the query.
    np.testing.assert_array_equal(
        np.asarray(jax.jit(lambda z: f(z) + (jnp.real(z) + 1))(x)), expected + 1)


def test_public_zero_breakpoint_routes_signed_subnormals_and_signed_zeros():
    words = np.asarray([0x8000000000000001, 0x8000000000000000, 0, 1], dtype=np.uint64)
    x = jax.lax.bitcast_convert_type(jnp.asarray(words), jnp.float64)
    f = _constant_pieces(0.0)
    expected = np.asarray([1.0, 7.0, 7.0, 2.0])
    np.testing.assert_array_equal(np.asarray(f(x)), expected)
    np.testing.assert_array_equal(np.asarray(jax.jit(lambda z: f(z))(x)), expected)


def test_public_complex_routing_uses_real_part_without_pointvalue_snap():
    real = np.asarray([0x8000000000000001, 0, 1], dtype=np.uint64).view(np.float64)
    z = np.zeros(3, dtype=np.complex128)
    z.real = real
    z.imag = 0.25
    f = _constant_pieces(0.0)
    expected = np.asarray([1.0, 2.0, 2.0])
    np.testing.assert_array_equal(np.asarray(f(jnp.asarray(z))), expected)
    np.testing.assert_array_equal(np.asarray(jax.jit(lambda x: f(x))(jnp.asarray(z))), expected)


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_public_native_half_open_nonfinite_routing(dtype):
    # columnFeval initializes zeros and uses [-inf, ..., inf) masks.
    # Thus NaN and +inf match no piece; -inf matches the first piece,
    # whose native Clenshaw arithmetic includes 0*inf and produces NaN.
    f = _constant_pieces(0.0)
    x = jnp.asarray(np.asarray([-np.inf, -2, -1, -0.5, 0, 0.5, 1, 2, np.inf, np.nan], dtype=dtype))
    expected = np.asarray([np.nan, 1, -3, 1, 7, 8, 4, 3, 0, 0])
    np.testing.assert_array_equal(np.asarray(f(x)), expected)
    np.testing.assert_array_equal(np.asarray(jax.jit(lambda z: f(z))(x)), expected)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_word_ordering_ieee_boundaries_and_nan(dtype):
    from chebfunjax.chebfun1d.chebfun import _exact_real_less, _exact_real_less_equal
    if dtype == np.float32:
        uint, sign, words = np.uint32, 0x80000000, [0, 1, 0x007fffff, 0x00800000, 0x3f800000, 0x7f7fffff, 0x7f800000, 0x7f800001, 0x7fc00000]
    else:
        uint, sign, words = np.uint64, 0x8000000000000000, [0, 1, 0x000fffffffffffff, 0x0010000000000000, 0x3ff0000000000000, 0x7fefffffffffffff, 0x7ff0000000000000, 0x7ff0000000000001, 0x7ff8000000000000]
    bits = np.asarray(words + [v | sign for v in words], dtype=uint)
    values = bits.view(dtype)
    x = jnp.asarray(values)
    # All pairs cover signed zeros, opposite signs, normal/subnormal boundary,
    # infinities and signaling/quiet NaNs. Expected comparisons run on host.
    with np.errstate(invalid='ignore'):
        for helper, expected in [(_exact_real_less, values[:, None] < values[None, :]),
                                 (_exact_real_less_equal, values[:, None] <= values[None, :])]:
            np.testing.assert_array_equal(
                np.asarray(jax.jit(helper)(x[:, None], x[None, :])), expected)


def test_static_piece_jit_scalar_matrix_and_traced_coefficients():
    f = _constant_pieces(0.0)
    for query in [jnp.asarray(-0.25), jnp.asarray([[-0.25, 0], [0.25, 0.75]])]:
        np.testing.assert_array_equal(np.asarray(jax.jit(lambda z: f(z))(query)), np.asarray(f(query)))
    # Trace coefficients with a fixed piece count and fixed intervals.
    def evaluate(coeffs, x):
        funs = [_Piece.from_coeffs(coeffs[i:i+1], a, b)
                for i, (a, b) in enumerate([(-1.0, 0.0), (0.0, 1.0)])]
        return Chebfun(funs=funs, domain=Domain((-1.0, 0.0, 1.0)))(x)
    np.testing.assert_array_equal(
        np.asarray(jax.jit(evaluate)(jnp.asarray([1., 2.]), jnp.asarray([-0.5, 0.5]))),
        np.asarray([1., 2.]))


@pytest.mark.parametrize('coefficients', [[1.0], [1.0, 2.0], [1.0, 2.0, 3.0]])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_bounded_single_piece_nonfinite_clenshaw_coordinates(coefficients, dtype):
    piece = _Piece.from_coeffs(jnp.asarray(coefficients), -1., 1.)
    f = Chebfun(funs=[piece], domain=Domain((-1., 1.)))
    x = jnp.asarray(np.asarray([-np.inf, np.inf, np.nan], dtype=dtype))
    assert np.all(np.isnan(np.asarray(f(x))))
    assert np.all(np.isnan(np.asarray(jax.jit(lambda z: f(z))(x))))
