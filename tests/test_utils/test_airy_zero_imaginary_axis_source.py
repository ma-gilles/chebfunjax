"""Exactly-real complex inputs retain the existing real Airy accuracy gate.

The eight-epsilon bound and120-digit data come unchanged from
 test_airy_real_phase.py; these are mathematical replacement controls,
not original MATLAB hermpts predicates or native trajectory tolerances.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.airy_general import airy_all


@pytest.mark.parametrize('imaginary_zero', [0., -0.])
def test_exact_float_highprecision_complex_negative_axis(imaginary_zero):
    rows = json.loads(Path(__file__).with_name('airy_real_phase_reference.json').read_text())['records']
    x = np.array([float.fromhex(r['x_hex']) for r in rows])
    z = x.astype(np.complex128)
    z.imag = imaginary_zero
    expected = np.array([[float(v) for v in r['reference120']] for r in rows])
    actual = np.asarray(jax.jit(airy_all)(jnp.asarray(z))).T
    amplitude = (-x)**.25 / np.sqrt(np.pi)
    inverse_amplitude = 1 / (np.sqrt(np.pi)*(-x)**.25)
    scales = np.stack((inverse_amplitude, amplitude, inverse_amplitude, amplitude), axis=1)
    assert actual.dtype == np.complex128
    assert np.all(np.abs(actual-expected) <= 8*np.finfo(float).eps*scales)


def test_mixed_complex_axis_tangent_and_nonfinite_contract():
    z = jnp.array([[-30.+0j, -1e3+0j], [2+3j, -10+17j]])
    values, tangent = jax.jit(lambda x: jax.jvp(airy_all, (x,), (1j*jnp.ones_like(x),)))(z)
    expected = (1j*values[1], 1j*z*values[0], 1j*values[3], 1j*z*values[2])
    for actual, reference in zip(tangent, expected):
        np.testing.assert_array_equal(actual, reference)
    real_values = airy_all(jnp.real(z[0]))
    for actual, reference in zip(values, real_values):
        np.testing.assert_array_equal(actual[0], reference.astype(jnp.complex128))
    assert all(a.shape == (0,) and a.dtype == jnp.complex128
               for a in airy_all(jnp.empty((0,), dtype=jnp.complex128)))
    assert np.isnan(np.asarray(airy_all(jnp.array([complex(np.inf, 0), complex(-np.inf, 0), complex(np.nan, 0)])))).all()
