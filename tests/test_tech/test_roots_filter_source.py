"""Native roots_main eigenvalue filtering at Chebfun7574c77.

Independent Python scalar predicates and stable numeric ordering implement the
literal source statements. Component-word assertions cover exact boundaries,
subnormal thresholds/order, zeros, nonfinite values and writable host storage.
The linear branch intentionally has its own different imaginary predicate.
"""
import json
import math
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech import chebtech as module


def _float(word):
    return struct.unpack('>d', struct.pack('>Q', word))[0]


def _words(values):
    return [struct.pack('>d', float(value)).hex() for value in values]


def _source(roots, tolerance):
    retained = [complex(z).real for z in roots
                if abs(complex(z).imag) < tolerance
                and abs(complex(z).real) <= 1.0+tolerance]
    retained.sort()
    if retained:
        retained[0] = max(retained[0], -1.)
        retained[-1] = min(retained[-1], 1.)
    return retained


_TINY = _float(1)
_REAL = [0., -0., _float(3), -_float(2), -_float(1), _float(1), -0., 0.,
         -1., 1., np.nextafter(1., 2.), np.nextafter(-1., -2.), 0.5, -0.5,
         math.inf, -math.inf, math.nan]
_TOLERANCES = [0., -0., -1., _TINY, _float(2), 10*np.finfo(float).eps, math.inf, math.nan]


@pytest.mark.parametrize('tolerance', _TOLERANCES, ids=lambda x: float(x).hex())
def test_eigenvalue_filter_literal_predicates_and_words(tolerance):
    imag = [0., -0., _TINY, -_TINY, _float(2), -_float(2), math.inf, math.nan]
    roots = np.asarray([complex(x, y) for x in _REAL for y in imag])
    expected = _source(roots, tolerance)
    actual = module._roots_filter_real(roots, tolerance)
    assert actual.dtype == np.float64 and actual.flags.writeable
    assert _words(actual) == _words(expected)
    fixed, count = module._roots_filter_real_jax(jnp.asarray(roots), tolerance)
    assert isinstance(fixed, jax.Array)
    assert int(count) == len(expected)
    assert _words(np.asarray(fixed)[:int(count)]) == _words(expected)


def test_eigenvalue_filter_strict_imaginary_and_inclusive_domain_boundary():
    tolerance = 8*np.finfo(float).eps
    inside_imag = np.nextafter(tolerance, 0.)
    outside_domain = np.nextafter(1.+tolerance, math.inf)
    roots = np.asarray([complex(.25, tolerance), complex(.5, inside_imag),
                        complex(1.+tolerance, 0.), complex(outside_domain, 0.),
                        complex(-1.-tolerance, 0.), complex(-outside_domain, 0.)])
    assert _words(module._roots_filter_real(roots, tolerance)) == _words([-1., .5, 1.])


def test_eigenvalue_filter_subnormal_order_and_zero_signs():
    roots = np.asarray([complex(x, 0.) for x in _REAL[:8]])
    expected = [-_float(2), -_float(1), 0., -0., -0., 0., _float(1), _float(3)]
    assert _words(module._roots_filter_real(roots, 1e-10)) == _words(expected)


def test_eigenvalue_filter_empty_spectrum():
    actual = module._roots_filter_real(np.asarray([], dtype=np.complex128), 1e-10)
    assert actual.shape == (0,) and actual.dtype == np.float64 and actual.flags.writeable


def test_real_leaf_driver_does_not_use_numpy_sort(monkeypatch):
    spectrum = np.asarray([.5, -.5], dtype=np.float64)
    # Compile before prohibiting NumPy's global function: JAX internals may
    # legitimately use NumPy for static shape metadata, outside library numerics.
    module._roots_filter_real(spectrum, 1e-10)
    monkeypatch.setattr(module, '_roots_default_eigenvalues', lambda coefficients: spectrum.copy())
    def forbidden(*args, **kwargs):
        raise AssertionError('host numeric sort reached')
    monkeypatch.setattr(np, 'sort', forbidden)
    actual = module._roots_main(np.asarray([0., 0., 1.]), 1e-10, recurse=False)
    assert _words(actual) == _words([-.5, .5])


@pytest.mark.parametrize('case', json.loads(
    (Path(__file__).parents[1]/'fixtures/roots_sort_native_20261018.json')
    .read_text())['cases'], ids=lambda row: row['name'])
def test_matlab_sort_signed_zero_and_tiny_words(case):
    words = case['input']['uint64_hex']
    values = np.asarray([_float(int(word, 16)) for word in words])
    roots = jax.lax.complex(jnp.asarray(values), jnp.zeros_like(jnp.asarray(values)))
    actual = module._roots_filter_real(roots, 1e-10)
    assert _words(actual) == [word.lower() for word in case['sorted']['uint64_hex']]


@pytest.mark.parametrize('length', [0, 1, 3, 15, 16, 17, 31, 32, 33, 49, 50, 51, 63, 64, 65, 127, 128, 129])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize('device_input', [False, True])
def test_filter_transfer_capacity_preserves_source_words(length, dtype, device_input):
    # Shape boundaries and rejected padding must not change accepted roots,
    # even with infinite/NaN tolerances or all-subnormal imaginary values.
    base = np.empty(len(_REAL)*3, dtype=dtype)
    base.real = np.tile(_REAL, 3)
    if np.issubdtype(dtype, np.complexfloating):
        real_dtype = np.float32 if dtype == np.complex64 else np.float64
        tiny = np.nextafter(real_dtype(0), real_dtype(1))
        base.imag = np.repeat([0., tiny, np.inf], len(_REAL))
    value = np.resize(base, length)
    argument = jnp.asarray(value) if device_input else value
    real_dtype = np.float32 if dtype in (np.float32, np.complex64) else np.float64
    tiny = np.nextafter(real_dtype(0), real_dtype(1))
    for tolerance in [0., -0., -1., float(tiny), float(2*tiny), 1e-10, math.inf, math.nan]:
        expected = np.asarray(_source(value, tolerance), dtype=real_dtype)
        actual = module._roots_filter_real(argument, tolerance)
        assert actual.dtype == expected.dtype and actual.flags.writeable
        assert actual.tobytes() == expected.tobytes()
