"""Port of MATLAB Chebfun tests/chebfun/test_extractColumns.m (Fable 5).

``extract_columns(cols)`` selects columns (0-based) of an array-valued
chebfun; a list may repeat/reorder columns, and an integer returns a scalar
chebfun.  The MATLAB pointValues comparisons have no counterpart (no
pointValues field) and are dropped.

Provenance
----------
MATLAB source : tests/chebfun/test_extractColumns.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(6178)
X = jnp.asarray(2 * RNG.uniform(size=100) - 1)


def _normest(f):
    return float(jnp.max(jnp.abs(f(X))))


class TestChebfunExtractcolumns:
    def test_extract_slice(self):
        # pass(1): extractColumns(f, 1:2) == [sin cos].
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = cj.chebfun(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1))
        g = cj.chebfun(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        h = f.extract_columns([0, 1])
        assert h.n_columns == 2 and _normest(g - h) < EPS

    def test_column_indexing(self):
        # pass(2): f(:, 1:2) via column indexing.
        pytest.skip("Chebfun.__getitem__ does not support 2-D column indexing f[:, cols]")

    def test_extract_repeat_reorder(self):
        # pass(3): extractColumns(f, [1 1 3 2]) == [sin sin exp cos].
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = cj.chebfun(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1))
        g = cj.chebfun(
            lambda x: jnp.stack([jnp.sin(x), jnp.sin(x), jnp.exp(x), jnp.cos(x)], axis=-1)
        )
        h = f.extract_columns([0, 0, 2, 1])
        assert h.n_columns == 4 and _normest(g - h) < 1e1 * EPS

    def test_unbounded(self):
        # pass(4): extractColumns on an unbounded domain.
        pytest.skip("chebfunjax has no unbounded-domain support")
