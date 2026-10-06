"""Fejer complex inverse DFT contracts with original strict sum bound.

Provenance
----------
MATLAB source : @chebtech1/quadwts.m; tests/chebtech/test_quadpts.m
Chebfun commit: 7574c77
Additional eager/JIT/moment checks supplement all original quadpts assertions.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.quadrature import chebweights


@pytest.mark.parametrize("compiled", [False, True])
def test_original_strict_sum_bound(compiled):
    weights = (jax.jit(lambda: chebweights(10, 1))() if compiled
               else chebweights(10, 1))
    # Original pass(m,1), including strict comparison.
    assert abs(float(jnp.sum(weights)) - 2) < 2 * np.finfo(float).eps


@pytest.mark.parametrize("n", [2, 9, 10, 11, 51])
def test_jit_preserves_quadrature_moments(n):
    eager = chebweights(n, 1)
    compiled = jax.jit(lambda: chebweights(n, 1))()
    # Independent polynomial moments; these extra JIT checks do not replace
    # any original MATLAB assertion or loosen its bound.
    nodes = np.cos(np.pi * (np.arange(n) + 0.5) / n)[::-1]
    for weights in (eager, compiled):
        assert weights.shape == (n,)
        assert bool(jnp.all(jnp.isfinite(weights)))
        for degree in range(min(n, 6)):
            integral = 0.0 if degree % 2 else 2.0 / (degree + 1)
            assert abs(float(weights @ jnp.asarray(nodes ** degree)) - integral) < 8 * np.finfo(float).eps
    np.testing.assert_allclose(compiled, eager, rtol=0, atol=4 * np.finfo(float).eps)
