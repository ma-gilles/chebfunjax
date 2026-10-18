"""Scaled QZ pencil and analytic roots, native @chebtech/roots.m7574c77.

Independent binary64 scalar arithmetic qualifies finite real pencil words.
Analytic polynomials qualify the actual JAX real/complex QZ paths with host
NumPy/SciPy eigenproviders prohibited. These are authored supplemental controls;
the unchanged original QZ tests retain their own native bounds.
"""
import struct

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from chebfunjax.tech import chebtech as module


@pytest.mark.parametrize('scale', [1., -1., 2.**-500, 2.**500])
def test_qz_real_pencil_literal_source_words(scale):
    coefficients = [scale * x for x in [0.375, -0.25, 0.125, 1.]]
    normalizer = max(abs(x) for x in coefficients)
    normalized = [x / normalizer for x in coefficients]
    column = [-0.5 * x for x in normalized[:-1]]
    column[-2] = column[-2] + 0.5 * normalized[-1]
    expected_a = [[0., 0.5, 0.], [0.5, 0., 1.], [0., 0.5, 0.]]
    for row, value in enumerate(reversed(column)):
        expected_a[row][0] = value
    expected_b = [[normalized[-1], 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    matrix, mass = module._roots_qz_pencil_jax(jnp.asarray(coefficients))
    assert isinstance(matrix, jax.Array) and isinstance(mass, jax.Array)
    for actual, expected in [(matrix, expected_a), (mass, expected_b)]:
        actual_words = np.asarray(actual).view(np.uint64).ravel().tolist()
        expected_words = [struct.unpack('>Q', struct.pack('>d', value))[0]
                          for row in expected for value in row]
        assert actual_words == expected_words


def _prohibit_host_eigenproviders(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('host eigensolver reached by QZ roots')
    for provider in [np.linalg, scipy.linalg]:
        for name in ['eig', 'eigvals']:
            monkeypatch.setattr(provider, name, forbidden)


@pytest.mark.parametrize('Tech', [module.Chebtech1, module.Chebtech2])
@pytest.mark.parametrize('complex_case', [False, True])
def test_qz_analytic_roots_without_host_eigenprovider(Tech, complex_case, monkeypatch):
    _prohibit_host_eigenproviders(monkeypatch)
    # T2(x) has roots +/-sqrt(1/2). For the complex control,
    # x**2-i/2 has roots +/-(1+i)/2, independently of any eigensolver.
    coefficients = ([0.5-0.5j, 0., 0.5] if complex_case else [0., 0., 1.])
    expected = (np.asarray([-0.5-0.5j, 0.5+0.5j]) if complex_case
                else np.asarray([-2.**-0.5, 2.**-0.5]))
    function = Tech.from_coeffs(jnp.asarray(coefficients))
    roots = np.asarray(function.roots(qz=True, all_roots=complex_case))
    assert roots.shape == (2,)
    roots = roots[np.argsort(roots.real)]
    assert np.max(np.abs(roots-expected)) < 10*np.finfo(float).eps
    assert np.max(np.abs(np.asarray(function(jnp.asarray(roots))))) < 10*np.finfo(float).eps
