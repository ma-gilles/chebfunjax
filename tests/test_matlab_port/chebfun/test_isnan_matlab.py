"""Port of MATLAB Chebfun tests/chebfun/test_isnan.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_isnan.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import chebfunjax as cj


class TestChebfunIsnan:
    def test_smooth_not_nan(self):
        f = cj.chebfun(lambda x: jnp.cos(2 * x))
        assert not bool(f.isnan())

    def test_empty_not_nan(self):
        # MATLAB pass(1): ~isnan(chebfun()).
        assert not bool(cj.chebfun().isnan())

    def test_piecewise_and_array_valued_not_nan(self):
        # MATLAB pass(2, 3).
        dom = [-1.0, -0.5, 0.0, 0.5, 1.0]
        f = cj.chebfun(jnp.sin, domain=dom)
        assert not bool(f.isnan())
        g = cj.chebfun(lambda x: jnp.stack(
            [jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1), domain=dom)
        assert not bool(g.isnan())

    def test_nan_point_values(self):
        # MATLAB pass(4): f.pointValues(2,1) = NaN makes isnan(f) true.
        import numpy as np
        dom = [-1.0, -0.5, 0.0, 0.5, 1.0]
        f = cj.chebfun(jnp.sin, domain=dom)
        pv = np.asarray(f.point_values).copy()
        pv[1] = np.nan
        assert bool(f.set_point_values(jnp.asarray(pv)).isnan())

    def test_nan_fun(self):
        # MATLAB pass(5): an artificially NaN-valued FUN.  chebfunjax's
        # adaptive constructor rejects NaN samples, so the NaN piece is
        # assembled directly, exactly as the MATLAB test does.
        from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
        from chebfunjax.domain import Domain
        dom = [-1.0, 0.0, 1.0]
        good = _Piece.from_function(jnp.sin, dom[0], dom[1])
        bad = _Piece.from_coeffs(
            jnp.asarray([jnp.nan], dtype=jnp.float64), dom[1], dom[2])
        f = Chebfun(funs=[good, bad], domain=Domain(tuple(dom)))
        assert bool(f.isnan())

    def test_nan_samples_rejected_by_constructor(self):
        # chebfunjax refuses to build a chebfun out of NaN samples rather
        # than silently producing a NaN representation.
        with pytest.raises(Exception):
            cj.chebfun(lambda x: jnp.full_like(x, jnp.nan))
