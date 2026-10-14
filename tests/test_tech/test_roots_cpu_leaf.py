"""Exact-source CPU leaf adapter controls; no tolerance changes."""
# uses-numpy: retained inputs, fixtures and independent byte comparisons.
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech import _roots_leaf_cpu as leaf
from chebfunjax.tech import chebtech as ct


def test_captured_leaves():
    archive = np.load(Path(__file__).parent / 'data/roots_cpu_leaf_inputs.npz')
    for key in archive.files:
        c = archive[key]
        n = len(c)
        padded = np.zeros(50)
        padded[:n] = c
        matrix, finite = leaf._matrix(jax.device_put(padded), jnp.asarray(n))
        expected = ct._roots_colleague_matrix_jax(jax.device_put(c))
        assert bool(finite)
        assert np.asarray(matrix)[:n-1, :n-1].tobytes() == np.asarray(expected).tobytes()
        a = ct._roots_default_eigenvalues(c)
        b = ct._roots_default_eigenvalues(jax.device_put(c))
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
        assert a.flags.owndata and a.flags.writeable


@pytest.mark.parametrize('c', [np.array([0., 0., 1.]), np.array([2., 0., 1.]),
    np.array([1., 0., 0.]), np.array([np.nan, 1., 1.]),
    np.array([np.inf, 1., 1.]), np.array([-np.inf, 1., 1.]),
    np.array([np.finfo(float).max, 0., np.finfo(float).tiny])])
def test_metadata_and_exact_solver_size(c, monkeypatch):
    calls = []
    original = ct._roots_eigvals_jax
    def observed(matrix):
        calls.append(matrix.shape)
        return original(matrix)
    monkeypatch.setattr(ct, '_roots_eigvals_jax', observed)
    baseline, finite = ct._roots_matrix_and_finite_jax(jax.device_put(c))
    if not bool(finite):
        with pytest.raises(np.linalg.LinAlgError, match='infs or NaNs'):
            ct._roots_default_eigenvalues(c)
        assert calls == []
    else:
        actual = ct._roots_default_eigenvalues(c)
        assert calls == [(len(c)-1, len(c)-1)]
        expected = ct._roots_default_eigenvalues(jax.device_put(c))
        assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('invalid', [complex(np.nan, 0), complex(np.inf, 0)])
def test_nonfinite_solver_output(invalid):
    def failed(matrix):
        return jnp.full((matrix.shape[0],), invalid, dtype=jnp.complex128)
    with pytest.raises(np.linalg.LinAlgError, match='did not converge'):
        leaf._eigenvalues(np.array([0., 0., 1.]), jax.devices('cpu')[0], failed)
