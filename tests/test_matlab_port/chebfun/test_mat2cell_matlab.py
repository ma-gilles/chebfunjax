"""Port of MATLAB Chebfun tests/chebfun/test_mat2cell.m (Fable 5).

``mat2cell([sizes])`` splits an array-valued chebfun column-wise into a list
of chebfuns whose column counts are ``sizes``.  MATLAB's zero-argument
``mat2cell(F)`` (split into single columns) maps to ``mat2cell([1, ...])``,
and the ``mat2cell(F, 1, [sizes])`` row-argument form maps to
``mat2cell([sizes])`` for a column chebfun.

The quasimatrix loop (k = 2), the row-chebfun (transpose) cases, and the
error-identifier cases have no chebfunjax counterpart and stay skipped.

Provenance
----------
MATLAB source : tests/chebfun/test_mat2cell.m
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
_DOM = (-1, -0.5, 0, 0.5, 1)


def _F():
    return cj.chebfun(
        lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1), domain=_DOM
    )


class TestChebfunMat2cell:
    def test_empty(self):
        # pass(k,1): mat2cell(chebfun()).
        from chebfunjax.chebfun1d.chebfun import chebfun
        C = chebfun().mat2cell()
        assert len(C) == 1 and C[0].isempty()

    def test_split_into_columns(self):
        # pass(k,2-5): mat2cell(F) -> {sin, cos, exp}, each a single column.
        # FIXED (Fable 5, Big-Three array-valued epic).
        F = _F()
        C = F.mat2cell([1, 1, 1])
        assert len(C) == 3 and all(c.n_columns == 1 for c in C)
        f = cj.chebfun(jnp.sin)
        g = cj.chebfun(jnp.cos)
        h = cj.chebfun(jnp.exp)
        assert float(jnp.max(jnp.abs(C[0](XR) - f(XR)))) < 10 * C[0].vscale * EPS
        assert float(jnp.max(jnp.abs(C[1](XR) - g(XR)))) < 10 * C[1].vscale * EPS
        assert float(jnp.max(jnp.abs(C[2](XR) - h(XR)))) < 10 * C[2].vscale * EPS

    def test_split_two_one(self):
        # pass(k,6-9): mat2cell(F, [2 1]) -> {[sin cos], exp}.
        # FIXED (Fable 5, Big-Three array-valued epic).
        F = _F()
        C = F.mat2cell([2, 1])
        assert len(C) == 2 and C[0].n_columns == 2 and C[1].n_columns == 1
        fg = cj.chebfun(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        h = cj.chebfun(jnp.exp)
        assert float(jnp.max(jnp.abs(C[0](XR) - fg(XR)))) < 10 * C[0].vscale * EPS
        assert float(jnp.max(jnp.abs(C[1](XR) - h(XR)))) < 10 * C[1].vscale * EPS

    def test_row_chebfuns(self):
        # pass(k,10-12): mat2cell of row chebfuns (F.') splits along the rows;
        # each cell is itself a row chebfun (isTransposed) with the requested
        # number of rows.
        Ft = _F().T

        # pass(10): mat2cell(Ft) -> 3 single-row cells.
        C = Ft.mat2cell()
        assert len(C) == 3
        assert all(c.is_transposed and c.size(1) == 1 for c in C)
        # each cell reproduces the corresponding component (sin, cos, exp).
        f = cj.chebfun(jnp.sin)
        g = cj.chebfun(jnp.cos)
        h = cj.chebfun(jnp.exp)
        for c, ref in zip(C, (f, g, h)):
            assert float(jnp.max(jnp.abs(c(XR) - ref(XR)))) < 10 * c.vscale * EPS

        # pass(11): mat2cell(Ft, [2 1]) -> cells with 2 and 1 rows.
        C = Ft.mat2cell([2, 1])
        assert len(C) == 2
        assert C[0].is_transposed and C[0].size(1) == 2
        assert C[1].is_transposed and C[1].size(1) == 1

    def test_error_conditions(self):
        # pass(k,13-17): mat2cell with invalid sizes raises.
        pytest.skip("Chebfun.mat2cell() does not validate that sizes sum to n_columns")

    def test_unbounded(self):
        # pass(k,18): mat2cell on an unbounded domain.
        pytest.skip("chebfunjax has no unbounded-domain support")
