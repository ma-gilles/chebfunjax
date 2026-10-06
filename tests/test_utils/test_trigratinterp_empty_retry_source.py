"""Source-empty coefficient conversion and retry error controls.

Provenance
----------
MATLAB source: trigratinterp.m, trig_rat_interp/getCoeffs,
sincosine_to_exponential and chopCoeffs.
Chebfun commit: 7574c77
These tests are controls for the source indexing behavior; they do not
qualify a full MATLAB retry with an empty numerator.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.utils import ratapprox as candidate


def test_empty_sincos_coefficients_preserve_source_a_one_index_error():
    with pytest.raises(IndexError):
        candidate._trigrat_sincos_to_exponential(jnp.empty((0,), dtype=jnp.float64))


def test_robust_retry_with_empty_numerator_reaches_source_index_error(monkeypatch):
    calls = []

    def controlled_svd(matrix, *, full_matrices):
        calls.append(matrix.shape)
        rows, cols = matrix.shape
        u = jnp.eye(rows, dtype=matrix.dtype)
        vh = jnp.eye(cols, dtype=matrix.dtype)
        if len(calls) == 1:
            # Three singular values are above threshold and two below, so the
            # source retry reduces n=1 to n=0. The right singular vector is
            # pure denominator, making the first chopped numerator empty.
            singular = jnp.asarray([1.0, 1.0, 1.0, 0.0, 0.0])
        else:
            singular = jnp.ones((min(rows, cols),), dtype=jnp.float64)
        return u, singular, vh

    monkeypatch.setattr(jnp.linalg, "svd", controlled_svd)
    nodes = jnp.linspace(-0.8, 0.8, 5)
    values = jnp.ones_like(nodes)
    with pytest.raises(IndexError):
        candidate._trigrat_fit_coefficients(
            values,
            1,
            1,
            nodes,
            False,
            False,
            robustness=True,
            interpolation=False,
            threshold=1e-12,
        )
    assert calls == [(5, 6), (5, 2)]
