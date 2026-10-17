"""Mathematical controls for the real oscillatory Airy specialization.

These exact-input references establish accuracy, not MATLAB builtin parity.
The eight-epsilon amplitude bound is an independent replacement-kernel gate;
all original source/native bounds are retained in their existing tests.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.utils.airy_general import airy_all


def test_exact_float_highprecision_real_oscillatory():
    rows = json.loads(Path(__file__).with_name('airy_real_phase_reference.json').read_text())['records']
    x = np.array([float.fromhex(r['x_hex']) for r in rows])
    expected = np.array([[float(v) for v in r['reference120']] for r in rows])
    actual = np.asarray(jax.jit(airy_all)(jnp.asarray(x))).T
    amplitude = (-x)**.25 / np.sqrt(np.pi)
    inverse_amplitude = 1 / (np.sqrt(np.pi)*(-x)**.25)
    scales = np.stack((inverse_amplitude, amplitude, inverse_amplitude, amplitude), axis=1)
    assert np.all(np.abs(actual-expected) <= 8*np.finfo(float).eps*scales)


def test_real_negative_analytic_tangents_and_ode():
    x = jnp.array([-np.nextafter(24., np.inf), -30., -1e3, -1e6, -1e8])
    values, tangent = jax.jit(lambda z: jax.jvp(airy_all, (z,), (jnp.ones_like(z),)))(x)
    expected = (values[1], x*values[0], values[3], x*values[2])
    for a, b in zip(tangent, expected):
        np.testing.assert_array_equal(a, b)
    second = jax.jit(lambda z: jax.jvp(
        lambda y: jax.jvp(airy_all, (y,), (jnp.ones_like(y),))[1],
        (z,), (jnp.ones_like(z),))[1])(x)
    expected_second = (x*values[0], values[0]+x*values[1],
                       x*values[2], values[2]+x*values[3])
    for a, b in zip(second, expected_second):
        np.testing.assert_allclose(a, b, rtol=8*np.finfo(float).eps, atol=0.)
