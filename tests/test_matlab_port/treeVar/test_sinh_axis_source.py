"""Captured sinh signs and analytic transverse derivatives.

Provenance
----------
MATLAB source: @treeVar/treeVar.m and @chebfun/sinh.m numeric dispatch.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
R2025b builtin scalar/array values are independently captured; its internal
implementation is opaque. Signed zero follows Re(sinh(x+iy))=sinh(x)*cos(y).
"""
import json
import math
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.treevar import _UNARY_FNS, _sinh_branch, to_first_order

DATA = Path(__file__).with_name('data')
ROWS = json.loads((DATA/'unary_storage_native36.json').read_text())['records']
ROWS += json.loads((DATA/'unary_storage_native_arrays.json').read_text())['records']
SINH_ROWS = [r for r in ROWS if r['method'] == 'sinh']


def decode(row):
    real = np.asarray([struct.unpack('>d', bytes.fromhex(v))[0] for v in row['real_hex']])
    imag = np.asarray([struct.unpack('>d', bytes.fromhex(v))[0] for v in row['imag_hex']])
    value = real if row['isreal'] else np.asarray(
        [complex(r, i) for r, i in zip(real, imag)], dtype=np.complex128)
    return value.reshape(row['shape'], order='F')


@pytest.mark.parametrize('row', SINH_ROWS, ids=lambda r:r['label'])
def test_native_sinh_values_storage_and_signs(row):
    expected = decode(row['output'])
    got = np.asarray(_UNARY_FNS['sinh'](jnp.asarray(decode(row['input']))))
    assert got.shape == expected.shape
    assert (not np.iscomplexobj(got)) == row['output']['isreal']
    for a, b in [(got.real, expected.real), (got.imag, expected.imag)]:
        assert np.array_equal(np.isnan(a), np.isnan(b))
        assert np.array_equal(np.isposinf(a), np.isposinf(b))
        assert np.array_equal(np.isneginf(a), np.isneginf(b))
        finite = np.isfinite(b)
        assert np.all(np.abs(a[finite]-b[finite]) <=
                      8*np.finfo(float).eps*np.maximum(1., np.abs(b[finite])))
        assert np.array_equal(a == 0, b == 0)
        zeros = b == 0
        assert np.array_equal(np.signbit(a[zeros]), np.signbit(b[zeros]))


@pytest.mark.parametrize('row', SINH_ROWS, ids=lambda r:r['label'])
def test_only_axis_zero_and_native_subnormal_imaginary_words_can_change(row):
    x = decode(row['input'])
    values = jnp.asarray(x)
    baseline = np.asarray(jnp.sinh(values))
    got = np.asarray(_sinh_branch(values))
    assert got.shape == baseline.shape and got.dtype == baseline.dtype
    # The qualified SINH repair intentionally retains tiny imaginary words
    # which the primitive loses. Require the captured native words there;
    # every other imaginary word must still equal this runtime's primitive.
    tiny = np.zeros(x.shape, dtype=bool)
    if x.dtype == np.dtype(np.complex128):
        bits = x.imag.copy().view(np.uint64) & np.uint64(0x7fffffffffffffff)
        with np.errstate(over='ignore', invalid='ignore'):
            finite_coefficient = np.isfinite(np.cosh(x.real))
        tiny = ((bits > 0) & (bits < np.uint64(0x0010000000000000))
                & np.isfinite(x.real) & finite_coefficient)
    expected = decode(row['output'])
    assert got.imag[tiny].tobytes() == expected.imag[tiny].tobytes()
    assert got.imag[~tiny].tobytes() == baseline.imag[~tiny].tobytes()
    if not np.iscomplexobj(x):
        assert got.tobytes() == baseline.tobytes()
        return
    on_axis = ((x.real.copy().view(np.uint64) & np.uint64(0x7fffffffffffffff)) == 0) & np.isfinite(x.imag)
    assert got.real[~on_axis].tobytes() == baseline.real[~on_axis].tobytes()
    assert np.all(got.real[on_axis] == 0)


@pytest.mark.parametrize('dtype', [jnp.complex64, jnp.complex128])
@pytest.mark.parametrize('x', [0., -0.])
@pytest.mark.parametrize('y', [1., -1., 2., -2.])
def test_jit_axis_sign_with_both_cosine_signs(dtype, x, y):
    value = jnp.asarray(complex(x, y), dtype=dtype)
    result = np.asarray(jax.jit(_UNARY_FNS['sinh'])(value))
    assert result.dtype == np.dtype(dtype)
    assert result.real == 0
    assert np.signbit(result.real) == (np.signbit(x) ^ np.signbit(math.cos(y)))


@pytest.mark.parametrize('dtype', [jnp.complex64, jnp.complex128])
@pytest.mark.parametrize('z', [.4+0j, 1j, -1j, 2j, -2j, .4+.3j])
def test_analytic_transverse_tangents(dtype, z):
    value = jnp.asarray(z, dtype=dtype)
    fn = _UNARY_FNS['sinh']
    expected_slope = complex(math.cosh(z.real)*math.cos(z.imag),
                             math.sinh(z.real)*math.sin(z.imag))
    eps = np.finfo(np.float32 if dtype == jnp.complex64 else np.float64).eps
    bound = 10*eps
    rhs = to_first_order(lambda u:u.diff()-u.sinh(), 0., (-1., 1.))[0]
    for tangent_value in (1., 1j):
        tangent = jnp.asarray(tangent_value, dtype=dtype)
        expected = jnp.asarray(expected_slope*tangent_value, dtype=dtype)
        for op in (fn, jax.jit(fn)):
            primal, derivative = jax.jvp(op, (value,), (tangent,))
            assert primal.dtype == dtype
            assert jnp.abs(derivative-expected) < bound
        for op in (rhs, jax.jit(rhs)):
            _, derivative = jax.jvp(lambda u:op(0., u),
                                    (jnp.asarray([value]),), (jnp.asarray([tangent]),))
            assert jnp.linalg.norm(derivative-expected) < bound


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_real_input_dtype_and_values_unchanged(dtype):
    x = jnp.asarray([-0., 0., -.4, .4, -jnp.inf, jnp.inf, jnp.nan], dtype=dtype)
    got = _UNARY_FNS['sinh'](x)
    expected = jnp.sinh(x)
    assert got.dtype == dtype
    assert np.asarray(got).tobytes() == np.asarray(expected).tobytes()
