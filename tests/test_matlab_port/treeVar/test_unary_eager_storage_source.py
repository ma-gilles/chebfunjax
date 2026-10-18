"""Native whole-array unary storage and traced complex tangent controls.

Provenance
----------
MATLAB source: @treeVar/treeVar.m numeric dispatch.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
R2025b scalar/array builtin captures establish storage behavior; builtin
arithmetic is opaque. JIT/AD checks are Python adapter contracts.
"""
import json
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.treevar import (
    _UNARY_FNS,
    _UNARY_FNS_RAW,
    _eager_unary_storage,
    to_first_order,
)

NAMES = 'asinh atan conj cos cosh exp expm1 sin sinh tan tanh uminus uplus'.split()
DATA = Path(__file__).with_name('data')
ROWS = json.loads((DATA/'unary_storage_native36.json').read_text())['records']
ROWS += json.loads((DATA/'unary_storage_native_arrays.json').read_text())['records']


def decode(row):
    re = np.array([struct.unpack('>d', bytes.fromhex(v))[0] for v in row['real_hex']])
    im = np.array([struct.unpack('>d', bytes.fromhex(v))[0] for v in row['imag_hex']])
    out = re if row['isreal'] else np.array(
        [complex(r, i) for r, i in zip(re, im)], dtype=np.complex128)
    return out.reshape(row['shape'], order='F')


@pytest.mark.parametrize('row', ROWS, ids=lambda r:r['method']+'_'+r['label'])
def test_primitive_component_bits_preserved(row):
    x = jnp.asarray(decode(row['input']))
    raw = np.asarray(_UNARY_FNS_RAW[row['method']](x))
    got = np.asarray(_UNARY_FNS[row['method']](x))
    assert got.shape == raw.shape
    assert got.real.tobytes() == raw.real.tobytes()
    if np.iscomplexobj(got):
        assert got.imag.tobytes() == raw.imag.tobytes()
    else:
        # Host integer classification here is an independent test oracle.
        assert np.all((raw.imag.copy().view(np.uint64) & np.uint64(0x7fffffffffffffff)) == 0)


@pytest.mark.parametrize('dtype', [jnp.complex64, jnp.complex128])
def test_storage_classifier_keeps_subnormal_imaginary_words(dtype):
    re_dtype = np.float32 if dtype == jnp.complex64 else np.float64
    tiny = np.nextafter(re_dtype(0), re_dtype(1))
    x = jnp.asarray(np.array([complex(.4, tiny)], dtype=np.complex64 if dtype == jnp.complex64 else np.complex128))
    out = _eager_unary_storage(lambda z:z)(x)
    assert out.dtype == dtype
    assert np.asarray(out).tobytes() == np.asarray(x).tobytes()


def slope(name, x):
    return {
        'asinh':lambda:1/jnp.sqrt(1+x*x), 'atan':lambda:1/(1+x*x),
        'conj':lambda:-1., 'cos':lambda:-jnp.sin(x), 'cosh':lambda:jnp.sinh(x),
        'exp':lambda:jnp.exp(x), 'expm1':lambda:jnp.exp(x),
        'sin':lambda:jnp.cos(x), 'sinh':lambda:jnp.cosh(x),
        'tan':lambda:1/jnp.cos(x)**2, 'tanh':lambda:1/jnp.cosh(x)**2,
        'uminus':lambda:-1., 'uplus':lambda:1.,
    }[name]()


@pytest.mark.parametrize('name', NAMES)
def test_real_axis_complex_tangents_and_rhs(name):
    x = jnp.asarray(.4+0j)
    tangent = jnp.asarray(.3j)
    fn = _UNARY_FNS[name]
    assert not jnp.iscomplexobj(fn(x))
    expected = tangent*slope(name, .4)
    for transformed in (fn, jax.jit(fn)):
        primal, derivative = jax.jvp(transformed, (x,), (tangent,))
        assert primal.dtype == jnp.complex128
        assert jnp.abs(derivative-expected) < 10*jnp.finfo(float).eps
    assert jax.vmap(fn)(jnp.asarray([.4+0j, -.4+0j])).dtype == jnp.complex128

    def op(u):
        value = -u if name == 'uminus' else +u if name == 'uplus' else getattr(u, name)()
        return u.diff()-value
    rhs = to_first_order(op, 0., (-1., 1.))[0]
    for transformed in (rhs, jax.jit(rhs)):
        primal, derivative = jax.jvp(lambda z:transformed(0., z),
                                    (jnp.asarray([x]),), (jnp.asarray([tangent]),))
        assert primal.dtype == jnp.complex128
        assert jnp.linalg.norm(derivative-expected) < 10*jnp.finfo(float).eps


OPEN = {(r['method'], r['label']): r['failures'] for r in json.loads(
    (DATA/'unary_storage_open_arithmetic.json').read_text())['records']}


@pytest.mark.parametrize('row', [
    pytest.param(r, marks=pytest.mark.xfail(strict=True, reason=
        'Unchanged primitive arithmetic: '+','.join(OPEN[(r['method'], r['label'])])))
    if (r['method'], r['label']) in OPEN else r for r in ROWS
], ids=lambda r:r['method']+'_'+r['label'])
def test_native_numeric_storage_and_components(row):
    assert row['ok']
    expected = decode(row['output'])
    got = np.asarray(_UNARY_FNS[row['method']](jnp.asarray(decode(row['input']))))
    assert got.shape == expected.shape
    assert (not np.iscomplexobj(got)) == row['output']['isreal']
    for value, target in [(got.real, expected.real), (got.imag, expected.imag)]:
        assert np.array_equal(np.isnan(value), np.isnan(target))
        assert np.array_equal(np.isposinf(value), np.isposinf(target))
        assert np.array_equal(np.isneginf(value), np.isneginf(target))
        finite = np.isfinite(target)
        assert np.all(np.abs(value[finite]-target[finite]) <=
                      8*np.finfo(float).eps*np.maximum(1., np.abs(target[finite])))
        zero = target == 0
        assert np.array_equal(value == 0, zero)
        assert np.all(value[zero] == 0)
        assert np.array_equal(np.signbit(value[zero]), np.signbit(target[zero]))


ASINH_AXIS = json.loads((DATA/'asinh_axis_native_20261018.json').read_text())


@pytest.mark.parametrize('case,row', list(zip(
    ASINH_AXIS['requested_inputs'], ASINH_AXIS['native_capture']['rows'])),
    ids=[c['name'] for c in ASINH_AXIS['requested_inputs']])
def test_asinh_native_axis_and_gradual_underflow(case, row):
    from chebfunjax.utils.elementary import _asinh_numeric

    def words(values):
        return np.asarray([int(v, 16) for v in values], dtype=np.uint64)

    x = np.empty(len(case['real']), np.complex128)
    x.real = words(case['real']).view(np.float64)
    x.imag = words(case['imag']).view(np.float64)
    expected = row['output']
    eager = np.asarray(_UNARY_FNS['asinh'](jnp.asarray(x)))
    assert (not np.iscomplexobj(eager)) == expected['isreal']
    for got in (eager, np.asarray(jax.jit(_asinh_numeric)(jnp.asarray(x)))):
        for part, values in [('real_words', got.real), ('imag_words', got.imag)]:
            target_words = words(expected[part])
            target = target_words.view(np.float64)
            actual_words = np.ascontiguousarray(values).view(np.uint64)
            assert np.array_equal(np.isnan(values), np.isnan(target))
            assert np.array_equal(np.isposinf(values), np.isposinf(target))
            assert np.array_equal(np.isneginf(values), np.isneginf(target))
            magnitude = target_words & np.uint64(0x7fffffffffffffff)
            # Check zero/subnormal values by integer words, independent of FTZ.
            tiny = magnitude < np.uint64(0x0010000000000000)
            assert np.array_equal(actual_words[tiny], target_words[tiny])
            normal = np.isfinite(target) & ~tiny
            assert np.all(np.abs(values[normal]-target[normal]) <=
                          8*np.finfo(float).eps*np.maximum(1., np.abs(target[normal])))


SIN_COS_AXIS_CAPTURE = json.loads(
    (DATA/'sin_cos_nonfinite_axis_native_20261018.json').read_text())
SIN_COS_AXIS_ROWS = [r for r in SIN_COS_AXIS_CAPTURE['capture']['rows']
                     if r['method'] in ('sin', 'cos')]
SIN_COS_NONFINITE_ROWS = [r for r in SIN_COS_AXIS_ROWS
                          if 'nonfinite' in r['name'] or
                          r['name'].startswith(('pinf_', 'ninf_', 'nan_'))]


def _decode_sin_cos_capture(row):
    value = np.empty(len(row['real']), dtype=np.complex128)
    value.real = np.array([int(v, 16) for v in row['real']], np.uint64).view(np.float64)
    value.imag = np.array([int(v, 16) for v in row['imag']], np.uint64).view(np.float64)
    return value.reshape(row['shape'], order='F')


@pytest.mark.parametrize('row', SIN_COS_NONFINITE_ROWS,
                         ids=lambda r:r['method']+'_'+r['name'])
def test_sin_cos_native_nonfinite_real_axis(row):
    """Fresh scalar/mixed native controls retain real NaNs and exact imag zeros."""
    value = jnp.asarray(_decode_sin_cos_capture(row['input']))
    target = _decode_sin_cos_capture(row['output'])
    eager = np.asarray(_UNARY_FNS[row['method']](value))
    assert (not np.iscomplexobj(eager)) == row['output']['isreal']
    for got in (eager, np.asarray(jax.jit(_UNARY_FNS_RAW[row['method']])(value))):
        assert got.shape == target.shape
        for actual, expected in ((got.real, target.real), (got.imag, target.imag)):
            assert np.array_equal(np.isnan(actual), np.isnan(expected))
            assert np.array_equal(np.isposinf(actual), np.isposinf(expected))
            assert np.array_equal(np.isneginf(actual), np.isneginf(expected))
            finite = np.isfinite(expected)
            assert np.all(np.abs(actual[finite]-expected[finite]) <=
                          8*np.finfo(float).eps*np.maximum(1., np.abs(expected[finite])))
            zero = expected == 0
            assert np.array_equal(actual == 0, zero)
            assert np.array_equal(np.signbit(actual[zero]), np.signbit(expected[zero]))


@pytest.mark.parametrize('row', SIN_COS_AXIS_ROWS,
                         ids=lambda r:r['method']+'_'+r['name'])
def test_sin_cos_only_nonfinite_axis_imaginary_component_changes(row):
    """All original real words and all out-of-scope imaginary words stay exact."""
    value = _decode_sin_cos_capture(row['input'])
    original = np.asarray(getattr(jnp, row['method'])(jnp.asarray(value)))
    got = np.asarray(_UNARY_FNS_RAW[row['method']](jnp.asarray(value)))
    assert got.shape == original.shape and got.dtype == original.dtype
    assert got.real.tobytes() == original.real.tobytes()
    zero_imag = (value.imag.copy().view(np.uint64) & np.uint64(0x7fffffffffffffff)) == 0
    selected = ~np.isfinite(value.real) & zero_imag
    assert got.imag[~selected].tobytes() == original.imag[~selected].tobytes()
    assert np.all(got.imag[selected].copy().view(np.uint64) == 0)


@pytest.mark.parametrize('name', ['sin', 'cos'])
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_sin_cos_nonzero_subnormal_imaginary_is_outside_axis_repair(name, dtype):
    real_dtype = np.float32 if dtype == np.complex64 else np.float64
    tiny = np.nextafter(real_dtype(0), real_dtype(1))
    value = np.empty(6, dtype=dtype)
    value.real = [np.inf, -np.inf, np.nan, np.inf, -np.inf, np.nan]
    value.imag = [tiny, tiny, tiny, -tiny, -tiny, -tiny]
    original = getattr(jnp, name)(jnp.asarray(value))
    for fn in (_UNARY_FNS_RAW[name], jax.jit(_UNARY_FNS_RAW[name])):
        got = fn(jnp.asarray(value))
        assert np.asarray(got).tobytes() == np.asarray(original).tobytes()


@pytest.mark.parametrize('name', ['sin', 'cos'])
@pytest.mark.parametrize('dtype', [jnp.complex64, jnp.complex128])
@pytest.mark.parametrize('z', [.4+0j, -.4-0j, 0j, 1j, -1j, .4+.3j])
def test_sin_cos_finite_transverse_jvp_retained(name, dtype, z):
    value = jnp.asarray(z, dtype=dtype)
    real_dtype = np.float32 if dtype == jnp.complex64 else np.float64
    for tangent_value in [1., 1j]:
        tangent = jnp.asarray(tangent_value, dtype=dtype)
        expected = jax.jvp(getattr(jnp, name), (value,), (tangent,))[1]
        for fn in (_UNARY_FNS[name], jax.jit(_UNARY_FNS[name])):
            _, got = jax.jvp(fn, (value,), (tangent,))
            assert jnp.abs(got-expected) <= 10*np.finfo(real_dtype).eps
