"""Native602FFT controls and independent linear-map/AD checks.

Provenance: Chebfun commit7574c77680d7e82b79626300bf255498271a72df,
@chebtech2/{coeffs2vals,vals2coeffs}.m. Fixture JSON binds neutral native
observations and archived input/output words; no candidate outputs are stored.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils._fft602 import complex_inverse602, real_forward602_values
from chebfunjax.utils.transforms import coeffs2vals, vals2coeffs

DATA = np.load(Path(__file__).parent/'fixtures/fft602_native.npz')
EPS = np.finfo(float).eps


def assert_words(actual, expected):
    actual = np.asarray(actual).copy()
    expected = np.asarray(expected).copy()
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize('kind', ['positive', 'independent'])
@pytest.mark.parametrize('layout', ['vector', 'matrix', 'tensor'])
def test_native_complex_inverse(kind, layout):
    x, expected = DATA['complex_'+kind], DATA['complex_'+kind+'_ifft']
    if layout == 'vector':
        x, expected = x[:,0], expected[:,0]
    elif layout == 'tensor':
        x, expected = x[:,None,:], expected[:,None,:]
    assert_words(complex_inverse602(jnp.asarray(x)), expected)


def test_native_real_forward():
    assert_words(real_forward602_values(jnp.asarray(DATA['real_positive'])),
                 DATA['real_positive_fft'])


def test_native_source_identity302_values():
    assert_words(coeffs2vals(jnp.eye(302,dtype=jnp.float64)), DATA['D'])


def test_native_source_complex_assembly():
    assert_words(vals2coeffs(jnp.asarray(DATA['V'])), DATA['Z'])


def test_real_and_pure_imaginary_storage_follow_real_source_branch():
    x = jnp.asarray(DATA['real_positive'][:302])
    real = vals2coeffs(x)
    np.testing.assert_array_equal(vals2coeffs(x.astype(jnp.complex128)), real)
    np.testing.assert_array_equal(vals2coeffs(1j*x), 1j*real)


def test_roundtrip302_real_and_complex():
    t = jnp.linspace(-1.,1.,302)
    for values in (jnp.exp(t), jnp.exp(t)+1j*jnp.sin(3*t)):
        recovered = coeffs2vals(vals2coeffs(values))
        np.testing.assert_allclose(recovered, values, rtol=0, atol=128*EPS)


def reference_complex_map(values):
    # Independent FFT expression of the mathematical linear DCT-I map.
    tmp = jnp.concatenate((values[:0:-1],values[:-1]))
    return jnp.fft.ifft(tmp)[:302].at[1:301].multiply(2.)


@pytest.mark.parametrize('storage', ['real', 'imaginary', 'mixed'])
def test_complex_transverse_jvp(storage):
    t = jnp.linspace(-.7,.8,302)
    x = t.astype(jnp.complex128)
    if storage == 'imaginary':
        x = 1j*x
    elif storage == 'mixed':
        x = x+1j*jnp.exp(t)
    direction = jnp.cos(t)+1j*jnp.sin(2*t)
    tangent = jax.jit(lambda x,v:jax.jvp(vals2coeffs,(x,),(v,))[1])(x,direction)
    expected = reference_complex_map(direction)
    np.testing.assert_allclose(tangent,expected,rtol=0,atol=128*EPS)


def test_complex_reverse_derivative():
    t = jnp.linspace(-.7,.8,302)
    x = t.astype(jnp.complex128)
    cotangent = jnp.sin(2*t)+1j*jnp.cos(t)
    _, pullback = jax.vjp(vals2coeffs,x)
    _, reference_pullback = jax.vjp(reference_complex_map,x)
    actual = jax.jit(pullback)(cotangent)[0]
    expected = reference_pullback(cotangent)[0]
    np.testing.assert_allclose(actual,expected,rtol=0,atol=256*EPS)


@pytest.mark.parametrize('storage', ['real5', 'imag5', 'complex_real5',
                                     'real1', 'imag1', 'complex_real1'])
def test_native_real_inverse_primal(storage):
    x = DATA['primal_'+storage+'_input']
    if storage.startswith('complex_real'):
        x = x.astype(np.complex128)
    expected = DATA['primal_'+storage+'_coeffs']
    # JAX preserves the statically complex input dtype. Compare both component
    # words to the native real result; dynamic output storage is not qualified.
    if storage.startswith('complex_real'):
        expected = expected.astype(np.complex128)
    assert_words(vals2coeffs(jnp.asarray(x)), expected)
