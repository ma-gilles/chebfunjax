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
    """Preserve words outside exact SIN zero inputs and require captured axis zeros."""
    value = _decode_sin_cos_capture(row['input'])
    original = np.asarray(getattr(jnp, row['method'])(jnp.asarray(value)))
    got = np.asarray(_UNARY_FNS_RAW[row['method']](jnp.asarray(value)))
    assert got.shape == original.shape and got.dtype == original.dtype
    zero_imag = (value.imag.copy().view(np.uint64) & np.uint64(0x7fffffffffffffff)) == 0
    expected_real = original.real.copy()
    if row['method'] == 'sin':
        zero_real = (value.real.copy().view(np.uint64) & np.uint64(0x7fffffffffffffff)) == 0
        expected_real[zero_real & zero_imag] = value.real[zero_real & zero_imag]
    assert got.real.tobytes() == expected_real.tobytes()
    selected = zero_imag
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


CONJ_UMINUS_ROWS = [r for r in SIN_COS_AXIS_CAPTURE['capture']['rows']
                    if r['method'] in ('conj', 'uminus')]


def _conj_uminus_primitive(name, value):
    return jnp.conj(value) if name == 'conj' else -value


@pytest.mark.parametrize('row', CONJ_UMINUS_ROWS,
                         ids=lambda r:r['method']+'_'+r['name'])
def test_conj_uminus_native_words_and_storage(row):
    """Existing requested-word capture independently covers all20 shapes/method."""
    value = jnp.asarray(_decode_sin_cos_capture(row['input']))
    expected = _decode_sin_cos_capture(row['output'])
    eager = np.asarray(_UNARY_FNS[row['method']](value))
    assert (not np.iscomplexobj(eager)) == row['output']['isreal']
    for got in (eager, np.asarray(jax.jit(_UNARY_FNS_RAW[row['method']])(value))):
        assert got.shape == expected.shape
        for actual, target in ((got.real, expected.real), (got.imag, expected.imag)):
            np.testing.assert_array_equal(np.ascontiguousarray(actual).view(np.uint64),
                                          np.ascontiguousarray(target).view(np.uint64))


@pytest.mark.parametrize('name', ['conj', 'uminus'])
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_conj_uminus_component_words_outside_zero_imaginary_unchanged(name, dtype):
    real_dtype = np.float32 if dtype == np.complex64 else np.float64
    word_dtype = np.uint32 if dtype == np.complex64 else np.uint64
    magnitude_mask = word_dtype(0x7fffffff if dtype == np.complex64 else 0x7fffffffffffffff)
    tiny = np.nextafter(real_dtype(0), real_dtype(1))
    value = np.empty(12, dtype=dtype)
    value.real = [0., -0., .4, -.4, tiny, -tiny, np.inf, -np.inf,
                  np.nan, -np.nan, .4, -.4]
    value.imag = [0., -0., tiny, -tiny, -.3, .3, np.inf, -np.inf,
                  np.nan, -np.nan, 0., -0.]
    baseline = np.asarray(_conj_uminus_primitive(name, jnp.asarray(value)))
    zero_imag = (value.imag.copy().view(word_dtype) & magnitude_mask) == 0
    for fn in (_UNARY_FNS_RAW[name], jax.jit(_UNARY_FNS_RAW[name])):
        got = np.asarray(fn(jnp.asarray(value)))
        assert got.dtype == value.dtype and got.shape == value.shape
        assert got.real.tobytes() == baseline.real.tobytes()
        assert got.imag[~zero_imag].tobytes() == baseline.imag[~zero_imag].tobytes()
        assert np.all(got.imag[zero_imag].copy().view(word_dtype) == 0)


@pytest.mark.parametrize('name', ['conj', 'uminus'])
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
@pytest.mark.parametrize('point', [(0., 0.), (-0., -0.), (.4, 0.),
                                   (-.4, -0.), (.4, .3), (-.4, -.3)])
def test_conj_uminus_independent_real_jacobian_and_hessian(name, dtype, point):
    """Ordinary real-linear maps have fixed independent2x2 Jacobians and zero Hessians."""
    def embedded(v):
        z = jax.lax.complex(v[0], v[1])
        out = _UNARY_FNS_RAW[name](z)
        return jnp.stack([jnp.real(out), jnp.imag(out)])
    value = jnp.asarray(point, dtype=dtype)
    expected = jnp.diag(jnp.asarray([1. if name == 'conj' else -1., -1.], dtype=dtype))
    for fn in (embedded, jax.jit(embedded)):
        np.testing.assert_array_equal(jax.jacfwd(fn)(value), expected)
        np.testing.assert_array_equal(jax.jacrev(fn)(value), expected)
        np.testing.assert_array_equal(jax.jacfwd(jax.jacfwd(fn))(value),
                                      jnp.zeros((2, 2, 2), dtype=dtype))
        for direction in [(1., 0.), (0., 1.), (.3, -.2)]:
            tangent = jnp.asarray(direction, dtype=dtype)
            _, got = jax.jvp(fn, (value,), (tangent,))
            np.testing.assert_array_equal(got, expected @ tangent)


@pytest.mark.parametrize('name', ['conj', 'uminus'])
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
@pytest.mark.parametrize('point', [(.4, 0.), (.4, .3)])
def test_conj_uminus_rhs_real_jacobian_and_hessian(name, dtype, point):
    op = (lambda u:u.diff()-u.conj()) if name == 'conj' else (lambda u:u.diff()-(-u))
    rhs = to_first_order(op, 0., (-1., 1.))[0]
    def embedded(v):
        z = jax.lax.complex(v[0], v[1])
        out = jnp.asarray(rhs(0., jnp.asarray([z]))).reshape(-1)[0]
        return jnp.stack([jnp.real(out), jnp.imag(out)])
    value = jnp.asarray(point, dtype=dtype)
    expected = jnp.diag(jnp.asarray([1. if name == 'conj' else -1., -1.], dtype=dtype))
    for fn in (embedded, jax.jit(embedded)):
        np.testing.assert_array_equal(jax.jacfwd(fn)(value), expected)
        np.testing.assert_array_equal(jax.jacrev(fn)(value), expected)
        np.testing.assert_array_equal(jax.jacfwd(jax.jacfwd(fn))(value),
                                      jnp.zeros((2, 2, 2), dtype=dtype))
        _, transverse = jax.jvp(fn, (value,), (jnp.asarray([0., 1.], dtype=dtype),))
        np.testing.assert_array_equal(transverse, expected[:, 1])


@pytest.mark.parametrize('name', ['conj', 'uminus'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_conj_uminus_real_input_words_and_dtype_unchanged(name, dtype):
    tiny = np.nextafter(dtype(0), dtype(1))
    value = jnp.asarray(np.array([0., -0., tiny, -tiny, .4, -.4,
                                 np.inf, -np.inf, np.nan, -np.nan], dtype=dtype))
    expected = np.asarray(_conj_uminus_primitive(name, value))
    for fn in (_UNARY_FNS_RAW[name], jax.jit(_UNARY_FNS_RAW[name])):
        got = np.asarray(fn(value))
        assert got.dtype == expected.dtype and got.tobytes() == expected.tobytes()


TAN_NONFINITE_ROWS = [r for r in ROWS if r['method'] == 'tan' and
                      r['label'] in ('complex_c_nonfinite_zero_imag',
                                     'complex_d_nonfinite_mixed')]


@pytest.mark.parametrize('row', TAN_NONFINITE_ROWS, ids=lambda r:r['label'])
def test_tan_nonfinite_axis_native_classes_storage_and_zero_signs(row):
    """Original immutable native records cover all-zero and mixed output storage."""
    value = jnp.asarray(decode(row['input']))
    expected = decode(row['output'])
    eager = np.asarray(_UNARY_FNS['tan'](value))
    assert (not np.iscomplexobj(eager)) == row['output']['isreal']
    for got in (eager, np.asarray(jax.jit(_UNARY_FNS_RAW['tan'])(value))):
        assert got.shape == expected.shape
        for actual, target in ((got.real, expected.real), (got.imag, expected.imag)):
            assert np.array_equal(np.isnan(actual), np.isnan(target))
            assert np.array_equal(np.isposinf(actual), np.isposinf(target))
            assert np.array_equal(np.isneginf(actual), np.isneginf(target))
            finite = np.isfinite(target)
            assert np.all(np.abs(actual[finite]-target[finite]) <=
                          8*np.finfo(float).eps*np.maximum(1., np.abs(target[finite])))
            zero = target == 0
            assert np.array_equal(actual == 0, zero)
            assert np.array_equal(np.signbit(actual[zero]), np.signbit(target[zero]))


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_tan_nonfinite_axis_only_selected_imaginary_words_change(dtype):
    real_dtype = np.float32 if dtype == np.complex64 else np.float64
    word_dtype = np.uint32 if dtype == np.complex64 else np.uint64
    magnitude_mask = word_dtype(0x7fffffff if dtype == np.complex64 else 0x7fffffffffffffff)
    tiny = np.nextafter(real_dtype(0), real_dtype(1))
    value = np.empty(16, dtype=dtype)
    value.real = [np.inf, -np.inf, np.nan, -np.nan, .4, -.4, 0., -0.,
                  np.inf, -np.inf, np.nan, -np.nan, .4, -.4, .4, -.4]
    value.imag = [0., -0., 0., -0., 0., -0., 0., -0.,
                  tiny, -tiny, tiny, -tiny, np.inf, -np.inf, np.nan, -np.nan]
    original = np.asarray(jnp.tan(jnp.asarray(value)))
    zero_imag = (value.imag.copy().view(word_dtype) & magnitude_mask) == 0
    selected = zero_imag
    expected_real = original.real.copy()
    zero_real = (value.real.copy().view(word_dtype) & magnitude_mask) == 0
    expected_real[zero_imag & zero_real] = value.real[zero_imag & zero_real]
    for fn in (_UNARY_FNS_RAW['tan'], jax.jit(_UNARY_FNS_RAW['tan'])):
        got = np.asarray(fn(jnp.asarray(value)))
        assert got.dtype == original.dtype and got.shape == original.shape
        assert got.real.tobytes() == expected_real.tobytes()
        assert got.imag[~selected].tobytes() == original.imag[~selected].tobytes()
        assert np.all(got.imag[selected].copy().view(word_dtype) == 0)


@pytest.mark.parametrize('dtype', [jnp.complex64, jnp.complex128])
@pytest.mark.parametrize('point', [0j, .4+0j, -.4-0j, 1j, -1j, .4+.3j])
def test_tan_nonfinite_axis_finite_first_second_derivatives_retained(dtype, point):
    import cmath

    value = jnp.asarray(point, dtype=dtype)
    reference = cmath.tan(point)
    first = 1+reference*reference
    second = 2*reference*first
    real_dtype = np.float32 if dtype == jnp.complex64 else np.float64
    eps = np.finfo(real_dtype).eps
    for fn in (_UNARY_FNS['tan'], jax.jit(_UNARY_FNS['tan'])):
        for direction in [1., 1j]:
            tangent = jnp.asarray(direction, dtype=dtype)
            _, actual_first = jax.jvp(fn, (value,), (tangent,))
            expected_first = first*direction
            assert jnp.abs(actual_first-expected_first) <= 10*eps*max(1., abs(expected_first))
            def derivative(z):
                return jax.jvp(fn, (z,), (tangent,))[1]
            _, actual_second = jax.jvp(derivative, (value,), (tangent,))
            expected_second = second*direction*direction
            assert jnp.abs(actual_second-expected_second) <= 10*eps*max(1., abs(expected_second))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_tan_nonfinite_axis_real_input_words_and_dtype_unchanged(dtype):
    tiny = np.nextafter(dtype(0), dtype(1))
    value = jnp.asarray(np.array([0., -0., tiny, -tiny, .4, -.4,
                                 np.inf, -np.inf, np.nan, -np.nan], dtype=dtype))
    expected = np.asarray(jnp.tan(value))
    for fn in (_UNARY_FNS_RAW['tan'], jax.jit(_UNARY_FNS_RAW['tan'])):
        got = np.asarray(fn(value))
        assert got.dtype == expected.dtype and got.tobytes() == expected.tobytes()


SIN_COS_FINITE_AXIS_ROWS = [r for r in SIN_COS_AXIS_ROWS
                            if r['name'] != 'complex_min_subnormal_imag']


@pytest.mark.parametrize('row', SIN_COS_FINITE_AXIS_ROWS,
                         ids=lambda r:r['method']+'_'+r['name'])
def test_sin_cos_finite_axis_native_zero_signs_and_storage(row):
    """Direct-component capture covers signed finite/nonfinite real-axis inputs."""
    value = jnp.asarray(_decode_sin_cos_capture(row['input']))
    expected = _decode_sin_cos_capture(row['output'])
    eager = np.asarray(_UNARY_FNS[row['method']](value))
    assert (not np.iscomplexobj(eager)) == row['output']['isreal']
    for got in (eager, np.asarray(jax.jit(_UNARY_FNS_RAW[row['method']])(value))):
        assert got.shape == expected.shape
        for component, actual, target in (('real', got.real, expected.real),
                                          ('imag', got.imag, expected.imag)):
            assert np.array_equal(np.isnan(actual), np.isnan(target))
            assert np.array_equal(np.isposinf(actual), np.isposinf(target))
            assert np.array_equal(np.isneginf(actual), np.isneginf(target))
            finite = np.isfinite(target)
            assert np.all(np.abs(actual[finite]-target[finite]) <=
                          8*np.finfo(float).eps*np.maximum(1., np.abs(target[finite])))
            zero = target == 0
            assert np.array_equal(actual == 0, zero)
            # Keep the distinct real-zero sign gap in a strict control below.
            if not (component == 'real' and row in SIN_COS_REAL_ZERO_GAP_ROWS):
                assert np.array_equal(np.signbit(actual[zero]), np.signbit(target[zero]))


@pytest.mark.parametrize('name', ['sin', 'cos'])
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_sin_cos_finite_axis_component_word_scope(dtype, name):
    real_dtype = np.float32 if dtype == np.complex64 else np.float64
    word_dtype = np.uint32 if dtype == np.complex64 else np.uint64
    mask = word_dtype(0x7fffffff if dtype == np.complex64 else 0x7fffffffffffffff)
    tiny = np.nextafter(real_dtype(0), real_dtype(1))
    value = np.empty(16, dtype=dtype)
    value.real = [0., -0., .4, -.4, 2., -2., np.inf, np.nan,
                  .4, -.4, np.inf, -np.inf, .4, -.4, .4, -.4]
    value.imag = [0., -0., 0., -0., 0., -0., 0., -0.,
                  tiny, -tiny, tiny, -tiny, .3, -.3, np.inf, np.nan]
    original = np.asarray(getattr(jnp, name)(jnp.asarray(value)))
    selected = (value.imag.copy().view(word_dtype) & mask) == 0
    expected_real = original.real.copy()
    if name == 'sin':
        zero_real = (value.real.copy().view(word_dtype) & mask) == 0
        expected_real[selected & zero_real] = value.real[selected & zero_real]
    for fn in (_UNARY_FNS_RAW[name], jax.jit(_UNARY_FNS_RAW[name])):
        got = np.asarray(fn(jnp.asarray(value)))
        assert got.dtype == original.dtype and got.shape == original.shape
        assert got.real.tobytes() == expected_real.tobytes()
        assert got.imag[~selected].tobytes() == original.imag[~selected].tobytes()
        assert np.all(got.imag[selected].copy().view(word_dtype) == 0)


def _sin_cos_independent_real_derivatives(name, point):
    import cmath

    z = complex(float(point[0]), float(point[1]))
    first = cmath.cos(z) if name == 'sin' else -cmath.sin(z)
    second = -cmath.sin(z) if name == 'sin' else -cmath.cos(z)
    jacobian = np.array([[first.real, -first.imag], [first.imag, first.real]])
    hessian = np.array([[[second.real, -second.imag], [-second.imag, -second.real]],
                        [[second.imag, second.real], [second.real, -second.imag]]])
    return jacobian, hessian


@pytest.mark.parametrize('name', ['sin', 'cos'])
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
@pytest.mark.parametrize('point', [(0., 0.), (-0., -0.), (.4, 0.), (-.4, -0.),
                                   (2., 0.), (-2., -0.), (.4, .3), (-.4, -.3)])
def test_sin_cos_finite_axis_independent_real_jacobian_hessian(name, dtype, point):
    def embedded(v):
        z = jax.lax.complex(v[0], v[1])
        out = _UNARY_FNS_RAW[name](z)
        return jnp.stack([jnp.real(out), jnp.imag(out)])
    value = jnp.asarray(point, dtype=dtype)
    expected_first, expected_second = _sin_cos_independent_real_derivatives(name, value)
    eps = np.finfo(np.float32 if dtype == jnp.float32 else np.float64).eps
    for fn in (embedded, jax.jit(embedded)):
        np.testing.assert_allclose(jax.jacfwd(fn)(value), expected_first, rtol=10*eps, atol=10*eps)
        np.testing.assert_allclose(jax.jacrev(fn)(value), expected_first, rtol=10*eps, atol=10*eps)
        np.testing.assert_allclose(jax.jacfwd(jax.jacfwd(fn))(value), expected_second,
                                   rtol=10*eps, atol=10*eps)
        for direction in [(1., 0.), (0., 1.)]:
            tangent = jnp.asarray(direction, dtype=dtype)
            _, got = jax.jvp(fn, (value,), (tangent,))
            np.testing.assert_allclose(got, expected_first @ np.array(direction),
                                       rtol=10*eps, atol=10*eps)


@pytest.mark.parametrize('name', ['sin', 'cos'])
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
@pytest.mark.parametrize('point', [(.4, 0.), (2., 0.)])
def test_sin_cos_finite_axis_rhs_real_jacobian_hessian(name, dtype, point):
    op = lambda u:u.diff()-getattr(u, name)()
    rhs = to_first_order(op, 0., (-1., 1.))[0]
    def embedded(v):
        z = jax.lax.complex(v[0], v[1])
        out = jnp.asarray(rhs(0., jnp.asarray([z]))).reshape(-1)[0]
        return jnp.stack([jnp.real(out), jnp.imag(out)])
    value = jnp.asarray(point, dtype=dtype)
    expected_first, expected_second = _sin_cos_independent_real_derivatives(name, value)
    eps = np.finfo(np.float32 if dtype == jnp.float32 else np.float64).eps
    for fn in (embedded, jax.jit(embedded)):
        np.testing.assert_allclose(jax.jacfwd(fn)(value), expected_first, rtol=10*eps, atol=10*eps)
        np.testing.assert_allclose(jax.jacfwd(jax.jacfwd(fn))(value), expected_second,
                                   rtol=10*eps, atol=10*eps)
        _, got = jax.jvp(fn, (value,), (jnp.asarray([0., 1.], dtype=dtype),))
        np.testing.assert_allclose(got, expected_first[:, 1], rtol=10*eps, atol=10*eps)


SIN_COS_REAL_ZERO_GAP_ROWS = [r for r in SIN_COS_FINITE_AXIS_ROWS
                             if r['method'] == 'sin' and r['name'] in
                             ('nzero_pzero', 'nzero_nzero', 'finite_axis_mixed_zero_signs')]


@pytest.mark.parametrize('row', SIN_COS_REAL_ZERO_GAP_ROWS, ids=lambda r:r['name'])
def test_sin_cos_finite_axis_native_real_zero_sign_gap(row):
    value = jnp.asarray(_decode_sin_cos_capture(row['input']))
    expected = _decode_sin_cos_capture(row['output'])
    eager = np.asarray(_UNARY_FNS[row['method']](value))
    for got in (eager, np.asarray(jax.jit(_UNARY_FNS_RAW[row['method']])(value))):
        actual, target = got.real, expected.real
        zero = target == 0
        assert np.array_equal(actual == 0, zero)
        assert np.array_equal(np.signbit(actual[zero]), np.signbit(target[zero]))


@pytest.mark.parametrize('name', ['sin', 'cos'])
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_sin_real_zero_boundary_word_scope(name, dtype):
    real_dtype = np.float32 if dtype == np.complex64 else np.float64
    word_dtype = np.uint32 if dtype == np.complex64 else np.uint64
    magnitude_mask = word_dtype(0x7fffffff if dtype == np.complex64 else 0x7fffffffffffffff)
    tiny = np.nextafter(real_dtype(0), real_dtype(1))
    value = np.empty(24, dtype=dtype)
    value.real = [0., -0., 0., -0., tiny, -tiny, tiny, -tiny,
                  0., -0., 0., -0., .4, -.4, 2., -2.,
                  np.inf, -np.inf, np.nan, -np.nan, .4, -.4, 0., -0.]
    value.imag = [0., 0., -0., -0., 0., -0., tiny, -tiny,
                  tiny, -tiny, .3, -.3, 0., -0., 0., -0.,
                  0., -0., 0., -0., .3, -.3, np.inf, np.nan]
    axis = (value.imag.copy().view(word_dtype) & magnitude_mask) == 0
    zero_real = (value.real.copy().view(word_dtype) & magnitude_mask) == 0
    argument = jnp.asarray(value)
    primitive = getattr(jnp, name)
    for original_fn, current_fn in ((primitive, _UNARY_FNS_RAW[name]),
                                    (jax.jit(primitive), jax.jit(_UNARY_FNS_RAW[name]))):
        expected = np.array(original_fn(argument), copy=True)
        expected.imag[axis] = real_dtype(0)
        if name == 'sin':
            expected.real[axis & zero_real] = value.real[axis & zero_real]
        got = np.asarray(current_fn(argument))
        assert got.dtype == expected.dtype
        assert got.real.tobytes() == expected.real.tobytes()
        assert got.imag.tobytes() == expected.imag.tobytes()


TAN_TANH_AXIS_CAPTURE = json.loads((DATA/'tan_tanh_axis_native_20261018.json').read_text())
TAN_TANH_OPEN_NAMES = {'unary_storage_native36.json__tan__complex_1_inf': 'Unresolved TAN infinite-imaginary phase/zero-sign convention', 'unary_storage_native_arrays.json__tan__complex_mixed': 'Unresolved TAN infinite-imaginary phase/zero-sign convention', 'unary_storage_native_arrays.json__tan__tan_cross_re_n1_im_pinf': 'Unresolved TAN infinite-imaginary phase/zero-sign convention', 'unary_storage_native_arrays.json__tan__tan_cross_re_p1_im_pinf': 'Unresolved TAN infinite-imaginary phase/zero-sign convention', 'unary_storage_native_arrays.json__tan__tan_cross_re_n2_im_pinf': 'Unresolved TAN infinite-imaginary phase/zero-sign convention', 'unary_storage_native_arrays.json__tan__tan_cross_re_p2_im_pinf': 'Unresolved TAN infinite-imaginary phase/zero-sign convention', 'unary_storage_native_arrays.json__tan__tan_cross_grid_6by4': 'Unresolved TAN infinite-imaginary phase/zero-sign convention', 'finite_imaginary_axis__tan': 'Independent fresh minimum-subnormal component-loss gap', 'finite_imaginary_axis__tanh': 'Independent fresh minimum-subnormal component-loss gap', 'near_axis_nonzero_words__tan': 'Independent fresh minimum-subnormal component-loss gap', 'near_axis_nonzero_words__tanh': 'Independent fresh minimum-subnormal component-loss gap', 'infinite_imaginary_phase_controls__tan': 'Unresolved TAN infinite-imaginary phase/zero-sign convention', 'infinite_real_controls__tanh': 'Fresh infinite-real TANH imaginary-zero sign gap'}
TAN_TANH_NATIVE_CASES = [pytest.param(r, id=r['name'], marks=(
    pytest.mark.xfail(strict=True, reason=TAN_TANH_OPEN_NAMES[r['name']])
    if r['name'] in TAN_TANH_OPEN_NAMES else [])) for r in TAN_TANH_AXIS_CAPTURE['rows']]


def _decode_tan_tanh_axis(v):
    a = np.empty(len(v['real']), dtype=np.complex128)
    a.real = np.array([int(h, 16) for h in v['real']], dtype=np.uint64).view(np.float64)
    a.imag = np.array([int(h, 16) for h in v['imag']], dtype=np.uint64).view(np.float64)
    return a.reshape(v['shape'], order='F')


@pytest.mark.parametrize('row', TAN_TANH_NATIVE_CASES)
def test_tan_tanh_axes_fresh_native_full_components(row):
    value = _decode_tan_tanh_axis(row['input'])
    value = value.real if row['input']['isreal'] else value
    argument = jnp.asarray(value)
    expected = _decode_tan_tanh_axis(row['output'])
    eager = np.asarray(_UNARY_FNS[row['method']](argument))
    assert (not np.iscomplexobj(eager)) == row['output']['isreal']
    for got in (eager, np.asarray(jax.jit(_UNARY_FNS_RAW[row['method']])(argument))):
        assert got.shape == expected.shape
        for actual, target in ((got.real, expected.real), (got.imag, expected.imag)):
            assert np.array_equal(np.isnan(actual), np.isnan(target))
            assert np.array_equal(np.isposinf(actual), np.isposinf(target))
            assert np.array_equal(np.isneginf(actual), np.isneginf(target))
            finite = np.isfinite(target)
            assert np.all(np.abs(actual[finite]-target[finite]) <=
                          8*np.finfo(float).eps*np.maximum(1., np.abs(target[finite])))
            zero = target == 0
            assert np.array_equal(actual == 0, zero)
            assert np.array_equal(np.signbit(actual[zero]), np.signbit(target[zero]))


def _tan_tanh_expected_axis_words(name, value, original):
    expected = np.array(original, copy=True)
    if not np.iscomplexobj(value):
        return expected
    word_dtype = np.uint32 if value.dtype == np.complex64 else np.uint64
    mask = word_dtype(0x7fffffff if value.dtype == np.complex64 else 0x7fffffffffffffff)
    zero_real = (value.real.copy().view(word_dtype) & mask) == 0
    zero_imag = (value.imag.copy().view(word_dtype) & mask) == 0
    if name == 'tan':
        expected.real[zero_real & zero_imag] = value.real[zero_real & zero_imag]
        expected.imag[zero_imag] = 0.
        imaginary_nan = zero_real & np.isnan(value.imag)
        expected.real[imaginary_nan] = np.nan
    else:
        selected = zero_real & ~zero_imag & np.isfinite(value.imag)
        expected.real[selected] = 0.
    return expected


@pytest.mark.parametrize('row', TAN_TANH_AXIS_CAPTURE['rows'], ids=lambda r:r['name'])
def test_tan_tanh_axes_capture_word_scope(row):
    value = _decode_tan_tanh_axis(row['input'])
    value = value.real if row['input']['isreal'] else value
    argument = jnp.asarray(value)
    primitive = getattr(jnp, row['method'])
    for original_fn, current_fn in ((primitive, _UNARY_FNS_RAW[row['method']]),
                                    (jax.jit(primitive), jax.jit(_UNARY_FNS_RAW[row['method']]))):
        expected = _tan_tanh_expected_axis_words(row['method'], value, np.asarray(original_fn(argument)))
        got = np.asarray(current_fn(argument))
        assert got.dtype == expected.dtype and got.shape == expected.shape
        assert got.real.tobytes() == expected.real.tobytes()
        assert got.imag.tobytes() == expected.imag.tobytes()


@pytest.mark.parametrize('name', ['tan', 'tanh'])
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_tan_tanh_axes_subnormal_and_origin_word_scope(name, dtype):
    real_dtype = np.float32 if dtype == np.complex64 else np.float64
    tiny = np.nextafter(real_dtype(0), real_dtype(1))
    normal = np.finfo(real_dtype).tiny
    value = np.empty(24, dtype=dtype)
    value.real = [0., -0., 0., -0., 0., -0., 0., -0., tiny, -tiny, normal, -normal,
                  .4, -.4, 0., -0., np.inf, -np.inf, np.nan, -np.nan, .4, -.4, 0., -0.]
    value.imag = [0., 0., -0., -0., tiny, -tiny, 1., -2., 0., -0., .3, -.3,
                  tiny, -tiny, np.inf, -np.inf, 0., -0., 0., -0., .3, -.3, np.nan, -np.nan]
    argument = jnp.asarray(value)
    primitive = getattr(jnp, name)
    for original_fn, current_fn in ((primitive, _UNARY_FNS_RAW[name]),
                                    (jax.jit(primitive), jax.jit(_UNARY_FNS_RAW[name]))):
        expected = _tan_tanh_expected_axis_words(name, value, np.asarray(original_fn(argument)))
        got = np.asarray(current_fn(argument))
        assert got.dtype == expected.dtype
        assert got.real.tobytes() == expected.real.tobytes()
        assert got.imag.tobytes() == expected.imag.tobytes()


def _tan_tanh_independent_derivatives(name, point):
    import cmath

    z = complex(float(point[0]), float(point[1]))
    f = cmath.tan(z) if name == 'tan' else cmath.tanh(z)
    first = 1+f*f if name == 'tan' else 1-f*f
    second = 2*f*first if name == 'tan' else -2*f*first
    jacobian = np.array([[first.real, -first.imag], [first.imag, first.real]])
    hessian = np.array([[[second.real, -second.imag], [-second.imag, -second.real]],
                        [[second.imag, second.real], [second.real, -second.imag]]])
    return jacobian, hessian


@pytest.mark.parametrize('name', ['tan', 'tanh'])
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
@pytest.mark.parametrize('point', [(0., 0.), (-0., -0.), (-0., 1.), (0., -1.),
                                   (-0., 2.), (0., -2.), (.4, 0.), (.4, .3)])
def test_tan_tanh_axes_independent_real_jacobian_hessian(name, dtype, point):
    def embedded(v):
        out = _UNARY_FNS_RAW[name](jax.lax.complex(v[0], v[1]))
        return jnp.stack([jnp.real(out), jnp.imag(out)])
    value = jnp.asarray(point, dtype=dtype)
    first, second = _tan_tanh_independent_derivatives(name, value)
    eps = np.finfo(np.float32 if dtype == jnp.float32 else np.float64).eps
    for fn in (embedded, jax.jit(embedded)):
        np.testing.assert_allclose(jax.jacfwd(fn)(value), first, rtol=10*eps, atol=10*eps)
        np.testing.assert_allclose(jax.jacrev(fn)(value), first, rtol=10*eps, atol=10*eps)
        np.testing.assert_allclose(jax.jacfwd(jax.jacfwd(fn))(value), second, rtol=10*eps, atol=10*eps)
        for direction in [(1., 0.), (0., 1.)]:
            _, got = jax.jvp(fn, (value,), (jnp.asarray(direction, dtype=dtype),))
            np.testing.assert_allclose(got, first @ np.array(direction), rtol=10*eps, atol=10*eps)


@pytest.mark.parametrize('name', ['tan', 'tanh'])
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
@pytest.mark.parametrize('point', [(-0., 1.), (-0., -2.)])
def test_tan_tanh_axes_rhs_real_jacobian_hessian(name, dtype, point):
    op = lambda u:u.diff()-getattr(u, name)()
    rhs = to_first_order(op, 0., (-1., 1.))[0]
    def embedded(v):
        out = jnp.asarray(rhs(0., jnp.asarray([jax.lax.complex(v[0], v[1])]))).reshape(-1)[0]
        return jnp.stack([jnp.real(out), jnp.imag(out)])
    value = jnp.asarray(point, dtype=dtype)
    first, second = _tan_tanh_independent_derivatives(name, value)
    eps = np.finfo(np.float32 if dtype == jnp.float32 else np.float64).eps
    for fn in (embedded, jax.jit(embedded)):
        np.testing.assert_allclose(jax.jacfwd(fn)(value), first, rtol=10*eps, atol=10*eps)
        np.testing.assert_allclose(jax.jacfwd(jax.jacfwd(fn))(value), second, rtol=10*eps, atol=10*eps)
        _, got = jax.jvp(fn, (value,), (jnp.asarray([0., 1.], dtype=dtype),))
        np.testing.assert_allclose(got, first[:, 1], rtol=10*eps, atol=10*eps)


@pytest.mark.parametrize('name', ['tan', 'tanh'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_tan_tanh_axes_real_input_primitive_words(name, dtype):
    value = jnp.asarray(np.array([-2., -.4, -0., 0., .4, 2., -np.inf, np.inf, -np.nan, np.nan], dtype=dtype))
    primitive = getattr(jnp, name)
    for original_fn, current_fn in ((primitive, _UNARY_FNS_RAW[name]),
                                    (jax.jit(primitive), jax.jit(_UNARY_FNS_RAW[name]))):
        expected = np.asarray(original_fn(value))
        got = np.asarray(current_fn(value))
        assert got.dtype == expected.dtype and got.tobytes() == expected.tobytes()


TAN_IMAGINARY_NAN_ROW = next(r for r in TAN_TANH_AXIS_CAPTURE['rows']
                            if r['name'] == 'nonfinite_imaginary_axis__tan')
TAN_IMAGINARY_NAN_INDICES = [i for i, (real, imag) in enumerate(zip(
    TAN_IMAGINARY_NAN_ROW['input']['real'], TAN_IMAGINARY_NAN_ROW['input']['imag']))
    if int(real, 16) & 0x7fffffffffffffff == 0 and int(imag, 16) & 0x7fffffffffffffff > 0x7ff0000000000000]


@pytest.mark.parametrize('index', TAN_IMAGINARY_NAN_INDICES)
def test_tan_imaginary_nan_direct_native_real_words_and_primitive_imaginary(index):
    row = TAN_IMAGINARY_NAN_ROW
    value = _decode_tan_tanh_axis(row['input'])
    argument = jnp.asarray(value)
    assert np.asarray(argument).real.tobytes() == value.real.tobytes()
    assert np.asarray(argument).imag.tobytes() == value.imag.tobytes()
    assert row['output']['real'][index] == '7ff8000000000000'
    for original_fn, current_fn in ((jnp.tan, _UNARY_FNS_RAW['tan']),
                                    (jax.jit(jnp.tan), jax.jit(_UNARY_FNS_RAW['tan'])),
                                    (jnp.tan, _UNARY_FNS['tan'])):
        original = np.asarray(original_fn(argument))
        got = np.asarray(current_fn(argument))
        actual = got.real.ravel(order='F')[index:index+1].copy().view(np.uint64)
        assert actual[0] == np.uint64(int(row['output']['real'][index], 16))
        assert got.imag.tobytes() == original.imag.tobytes()


def _tan_imaginary_nan_word_context(dtype, cohort, shape):
    real_dtype = np.float32 if dtype == np.complex64 else np.float64
    word_dtype = np.uint32 if dtype == np.complex64 else np.uint64
    tiny = np.nextafter(real_dtype(0), real_dtype(1))
    positive = word_dtype(0x7fc00021 if dtype == np.complex64 else 0x7ff8000000000021)
    negative = word_dtype(0xffc01234 if dtype == np.complex64 else 0xfff8000000001234)
    payloads = np.array([positive, negative], dtype=word_dtype).view(real_dtype)
    value = np.empty(12, dtype=dtype)
    value.real[:4] = [0., -0., 0., -0.]
    value.imag[:4] = [np.nan, np.nan, -np.nan, -np.nan] if cohort != 1 else [*payloads, *payloads]
    if cohort == 0:
        value.real[4:] = [-tiny, tiny, -1., 1., -np.nan, np.nan, -0., 0.]
        value.imag[4:] = [np.nan, -np.nan, np.nan, -np.nan, .3, -.3, np.inf, -np.inf]
    elif cohort == 1:
        value.real[4:] = [-tiny, tiny, *payloads, *payloads, -0., 0.]
        value.imag[4:] = [*payloads, 0., -0., .3, -.3, np.inf, -np.inf]
    else:
        value.real[4:] = [-2., -1., 1., 2., -.4, .4, -0., 0.]
        value.imag[4:] = [np.inf, -np.inf, np.inf, -np.inf, 0., -0., np.inf, -np.inf]
    shapes = [(12,), (1, 12), (3, 4), (12, 1)]
    return value.reshape(shapes[shape], order='F')


TAN_IMAGINARY_NAN_PRIOR_CAPTURE = json.loads((DATA/'tan_imaginary_nan_prior_words_20261018.json').read_text())
TAN_IMAGINARY_NAN_PRIOR_WORDS = {r['label']:r for r in TAN_IMAGINARY_NAN_PRIOR_CAPTURE['rows']}


def _decode_tan_nan_prior_words(row, mode):
    dtype = np.dtype(row['dtype'])
    word_dtype = np.uint32 if dtype == np.complex64 else np.uint64
    real_dtype = np.float32 if dtype == np.complex64 else np.float64
    out = np.empty(len(row[mode]['real']), dtype=dtype)
    out.real = np.array([int(v, 16) for v in row[mode]['real']], dtype=word_dtype).view(real_dtype)
    out.imag = np.array([int(v, 16) for v in row[mode]['imag']], dtype=word_dtype).view(real_dtype)
    return out.reshape(row['shape'])


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('cohort', [0, 1, 2])
@pytest.mark.parametrize('shape', [0, 1, 2, 3])
def test_tan_imaginary_nan_signed_payload_and_offaxis_word_scope(dtype, cohort, shape):
    value = _tan_imaginary_nan_word_context(dtype, cohort, shape)
    argument = jnp.asarray(value)
    assert np.asarray(argument).real.tobytes() == value.real.tobytes()
    assert np.asarray(argument).imag.tobytes() == value.imag.tobytes()
    row = TAN_IMAGINARY_NAN_PRIOR_WORDS[f'{np.dtype(dtype).name}_{cohort}_{shape}']
    assert row['shape'] == list(value.shape)
    input_value = _decode_tan_nan_prior_words(row, 'input')
    assert input_value.real.tobytes() == value.real.tobytes()
    assert input_value.imag.tobytes() == value.imag.tobytes()
    for mode, current_fn in (('eager', _UNARY_FNS_RAW['tan']),
                              ('jit', jax.jit(_UNARY_FNS_RAW['tan']))):
        prior = _decode_tan_nan_prior_words(row, mode)
        expected = _tan_tanh_expected_axis_words('tan', value, prior)
        got = np.asarray(current_fn(argument))
        assert got.dtype == expected.dtype and got.shape == expected.shape
        assert got.real.tobytes() == expected.real.tobytes()
        assert got.imag.tobytes() == expected.imag.tobytes()
