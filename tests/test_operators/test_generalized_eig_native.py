"""Native dense eig dispatch/range contract, Chebfun 7574c77.

Fixtures contain MATLAB R2025b outputs and raw input words, including storage.
The provider never reads these fixtures. NaN payloads are not a numeric contract;
scalar finite vector and infinity component words are checked exactly.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.blocklinop import _geig

FIXTURE = json.loads((Path(__file__).parent / 'fixtures' /
                      'generalized_eig_native.json').read_text())


def decode(record):
    """Assign components separately: complex multiplication can corrupt Inf/NaN."""
    real = np.array([int(w, 16) for w in record['real_words']], np.uint64).view(np.float64)
    imag = np.array([int(w, 16) for w in record['imag_words']], np.uint64).view(np.float64)
    if record['isreal']:
        assert not np.any(imag)
        return real.reshape(record['shape'], order='F')
    result = np.empty(real.shape, np.complex128)
    result.real, result.imag = real, imag
    return result.reshape(record['shape'], order='F')


def component_words(values):
    return np.ascontiguousarray(values, dtype=np.float64).ravel().view(np.uint64)


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda case: case['id'])
def test_native_dispatch_and_range(case):
    A, B = decode(case['A']), decode(case['B'])
    actual, vectors = _geig(jnp.asarray(A), jnp.asarray(B))
    actual, vectors = np.asarray(actual), np.asarray(vectors)
    expected = decode(case['eigenvalues']).ravel()
    assert actual.shape == (len(A),)
    assert vectors.shape == A.shape
    for got, want in [(actual.real, expected.real), (actual.imag, expected.imag)]:
        # Preserve native order, infinities and 0:0 imaginary zero.
        np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
        np.testing.assert_array_equal(np.isposinf(got), np.isposinf(want))
        np.testing.assert_array_equal(np.isneginf(got), np.isneginf(want))
        zero = want == 0
        np.testing.assert_array_equal(got[zero], want[zero])
        finite = np.isfinite(want) & ~zero
        np.testing.assert_allclose(got[finite], want[finite], rtol=1e-12, atol=0)
    if case['exact_components']:
        assert (not np.iscomplexobj(vectors)) == case['vectors']['isreal']
        assert (not np.iscomplexobj(actual)) == case['eigenvalues']['isreal']
        reference_vectors = decode(case['vectors'])
        for got, want in [(actual.real, expected.real), (actual.imag, expected.imag),
                          (vectors.real, reference_vectors.real),
                          (vectors.imag, reference_vectors.imag)]:
            np.testing.assert_array_equal(component_words(got), component_words(want))
    if case['b_normalized']:
        assert np.linalg.norm(vectors.conj().T @ B @ vectors - np.eye(len(A))) < 2e-14
    # Standard-range captures also require eigenvectors to solve their pencil.
    # Extreme scaling is checked through native spectra and B-normalization above.
    if case['source'] == 'dispatch' and np.all(np.isfinite(actual)):
        error = np.linalg.norm(A @ vectors - (B @ vectors) * actual)
        bound = (np.linalg.norm(A) + np.linalg.norm(B) * np.max(abs(actual))) * np.linalg.norm(vectors)
        assert error / bound < 2e-14
