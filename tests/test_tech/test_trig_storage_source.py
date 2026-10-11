"""Storage boundaries captured from native Chebfun 7574c77."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech, trig_vals2coeffs
from chebfunjax.utils._polynomial_roots_source import polynomial_roots

ROWS = json.loads(Path(__file__).with_name('trig_storage_native.json').read_text())['rows']


def array(record):
    value = np.asarray(record['real']).reshape(record['size'])
    if not record['isreal']:
        value = value.astype(complex) + 1j*np.asarray(record['imag']).reshape(record['size'])
    return jnp.asarray(value[:, 0] if value.shape[1] == 1 else value)


def check(actual, record):
    expected = array(record)
    assert jnp.iscomplexobj(actual) == (not record['isreal'])
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('row', ROWS, ids=[row['label'] for row in ROWS])
def test_native_storage_boundaries(row):
    value = array(row['input'])
    if row['label'].startswith('coeff_'):
        f = Trigtech.from_coeffs(value)
    else:
        check(trig_vals2coeffs(value), row['directTransform'])
        f = Trigtech.from_values(value)
    check(f.coeffs, row['coeffs'])
    check(f.prolong(7).coeffs, row['prolonged'])
    check(f.simplify().coeffs, row['simplified'])
    if row['complex_roots']:
        np.testing.assert_array_equal(f.roots(complex=True), array(row['complex_roots']))


@pytest.mark.parametrize('complex_input', [False, True])
def test_traced_transform_keeps_values_and_constant_storage(complex_input):
    dtype = jnp.complex128 if complex_input else jnp.float64
    values = jnp.asarray([1., 2., 3., 2.], dtype=dtype)
    np.testing.assert_array_equal(jax.jit(trig_vals2coeffs)(values), trig_vals2coeffs(values))
    constant = jnp.asarray([2.], dtype=dtype)
    assert jax.jit(trig_vals2coeffs)(constant).dtype == constant.dtype


def test_polynomial_explicit_complex_zero_remains_complex_dispatch(monkeypatch):
    seen = []
    eigvals = jnp.linalg.eigvals

    def observe(matrix):
        seen.append(matrix.dtype)
        return eigvals(matrix)

    monkeypatch.setattr(jnp.linalg, "eigvals", observe)
    polynomial_roots(jnp.asarray([.5, 2., .5]))
    polynomial_roots(jnp.asarray([.5, 2., .5], dtype=jnp.complex128))
    assert seen == [jnp.dtype("float64"), jnp.dtype("complex128")]
