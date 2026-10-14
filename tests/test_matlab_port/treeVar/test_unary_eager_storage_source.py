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
