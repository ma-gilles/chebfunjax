"""Pinned MATLAB Clenshaw values: real source bits and generic complex bounds.

Capture: Chebfun 7574c77680d7e82b79626300bf255498271a72df, MATLAB R2025b.
The MAT reader intentionally uses mat_dtype=False: SciPy mat_dtype=True can
silently discard imaginary components in complex fields in this fixture.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.io import loadmat

from chebfunjax.tech.chebtech import Chebtech2, _clenshaw

_DATA = loadmat(Path(__file__).parents[1] / 'references/clenshaw_native/native_clenshaw.mat',
                mat_dtype=False, struct_as_record=False)
_REAL = [c.item() for c in _DATA['cases'].ravel()]
_COMPLEX = [c.item() for c in _DATA['complex_cases'].ravel()]


def _bits(a, b):
    np.testing.assert_array_equal(np.asarray(a).view(np.uint64), np.asarray(b).view(np.uint64))


@pytest.mark.parametrize('case', _REAL, ids=lambda c: str(c.name.item()))
def test_native_real_source_bits(case):
    coeff = case.coeff[:, 0] if case.coeff.shape[1] == 1 else case.coeff
    x = case.x[:, 0]
    expected = case.values[:, 0] if coeff.ndim == 1 else case.values
    _bits(_clenshaw(jnp.asarray(coeff), jnp.asarray(x)), expected)
    scalar_expected = case.scalar_value.reshape(()) if coeff.ndim == 1 else case.scalar_value[0]
    _bits(Chebtech2.from_coeffs(jnp.asarray(coeff))(float(case.scalar_x.item())), scalar_expected)


@pytest.mark.parametrize('case', _COMPLEX, ids=lambda c: str(c.name.item()))
def test_native_complex_generic_bound(case):
    assert np.iscomplexobj(case.coeff) and np.any(case.coeff.imag != 0)
    actual = _clenshaw(jnp.asarray(case.coeff[:, 0]), jnp.asarray(case.x[:, 0]))
    expected = case.values[:, 0]
    np.testing.assert_allclose(actual, expected, rtol=0,
                               atol=10*np.finfo(float).eps*max(1., np.max(np.abs(expected))))


def test_native_chemical_tech_and_public_endpoint_values():
    import chebfunjax as cj

    case = _DATA['chemical'].item()
    c = jnp.asarray(case.coeff)
    _bits(_clenshaw(c, jnp.asarray(case.canonical_x[:, 0])), case.values)
    f = cj.Chebfun.from_coeffs(c).set_point_values(jnp.asarray(case.point_values))
    x = jnp.asarray(case.physical_x[:, 0])
    _bits(f(x), case.public_initial)
    _bits(jax.jit(lambda z: f(z))(x), case.public_initial)
