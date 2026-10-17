"""Analytic and resource controls for the source AAA/Lawson SVD.

The standard normwise budget scales with matrix size and binary64 epsilon;
no componentwise relative accuracy of tiny singular vectors is assumed.
"""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

minimax_module = importlib.import_module('chebfunjax.utils.minimax')


@pytest.mark.parametrize('scale', [1., 1e-100, 1e-150])
def test_lawson_svd_uniform_scale_backward_error(scale):
    base = np.array([[1., 0., 1.], [0., 1., 2.], [0., 0., 0.],
                     [0., 0., 0.], [0., 0., 0.]])
    matrix = scale*base
    u, singular, vh = map(np.asarray, minimax_module._lawson_svd(jnp.asarray(matrix)))
    budget = 32*max(matrix.shape)*np.finfo(float).eps
    # Exact nonzero singular values are sqrt(6) and 1 for this rank-two matrix.
    np.testing.assert_allclose(singular/scale, [np.sqrt(6.), 1., 0.],
                               rtol=budget, atol=budget)
    norm = np.linalg.norm(matrix, ord=np.inf)
    assert np.linalg.norm((u*singular)@vh-matrix, ord=np.inf) <= budget*norm
    assert np.linalg.norm(vh@vh.T-np.eye(3), ord=np.inf) <= budget
    assert np.linalg.norm(u.T@u-np.eye(3), ord=np.inf) <= budget


def test_lawson_outer_jit_avoids_host_svd(monkeypatch):
    matrix = jnp.asarray([[1., 0., 1e-100], [0., 1., 2e-100], [0., 0., 0.]])
    def forbidden(*args, **kwargs):
        raise AssertionError('AAA/Lawson SVD must execute in JAX')
    monkeypatch.setattr(np.linalg, 'svd', forbidden)
    u, singular, vh = map(np.asarray, jax.jit(minimax_module._lawson_svd)(matrix))
    budget = 32*max(matrix.shape)*np.finfo(float).eps
    error = np.linalg.norm((u*singular)@vh-np.asarray(matrix), ord=np.inf)
    assert error <= budget*np.linalg.norm(np.asarray(matrix), ord=np.inf)
