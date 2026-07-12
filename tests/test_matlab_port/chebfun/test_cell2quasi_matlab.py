"""Port of MATLAB Chebfun tests/chebfun/test_cell2quasi.m (Fable 5).

FIXED: cell2quasi added in the Fable 5 audit (list of chebfuns ->
Quasimatrix, the chebfunjax array-valued counterpart).

Provenance
----------
MATLAB source : tests/chebfun/test_cell2quasi.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

XS = jnp.asarray(np.linspace(-0.95, 0.95, 20))


class TestChebfunCell2quasi:
    def test_roundtrip(self):
        f = cj.chebfun(jnp.sin)
        g = cj.chebfun(jnp.cos)
        Q = cj.cell2quasi([f, g])
        assert Q.n_cols == 2
        assert float(jnp.max(jnp.abs(Q.cols[0](XS) - f(XS)))) == 0.0
        assert float(jnp.max(jnp.abs(Q.cols[1](XS) - g(XS)))) == 0.0

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            cj.cell2quasi([])
