"""Port of MATLAB Chebfun tests/chebfun/test_repmat.m (Fable 5).

``repmat(k)`` tiles a scalar chebfun into an array-valued chebfun with ``k``
identical columns.  The vertical-tiling case (repmat of a row chebfun) is
commented out in the MATLAB source (needs vertcat) and stays skipped.

Provenance
----------
MATLAB source : tests/chebfun/test_repmat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(7681)
XR = jnp.asarray(2 * RNG.uniform(size=1000) - 1)


class TestChebfunRepmat:
    def test_repmat_three_columns(self):
        # pass(1, 2): repmat(f, 1, 3) == [sin sin sin].
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = cj.chebfun(jnp.sin, domain=(-1, 0, 1))
        Q = f.repmat(3)
        exact = jnp.stack([jnp.sin(XR), jnp.sin(XR), jnp.sin(XR)], axis=-1)
        assert Q.n_columns == 3
        assert float(jnp.max(jnp.abs(Q(XR) - exact))) < 10 * Q.vscale * EPS

    def test_vertical_tiling(self):
        # pass(3): repmat(f.', 3, 1) -- commented out in MATLAB (needs vertcat).
        pytest.skip("chebfunjax has no row-chebfun transpose / vertcat")
