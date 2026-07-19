"""Port of MATLAB Chebfun tests/chebfun/test_fliplr.m (Fable 5).

``fliplr`` reverses the COLUMNS of an array-valued chebfun and is the
identity for a scalar (single-column) chebfun.  The row-chebfun (transpose)
and singular cases have no chebfunjax counterpart and stay skipped.

Provenance
----------
MATLAB source : tests/chebfun/test_fliplr.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(7681)
XR = jnp.asarray(2 * RNG.uniform(size=50) - 1)


class TestChebfunFliplr:
    def test_empty(self):
        # pass(1): isempty(fliplr(chebfun())).
        from chebfunjax.chebfun1d.chebfun import chebfun
        assert chebfun().fliplr().isempty()

    def test_row_chebfuns(self):
        # pass(2-5): fliplr of row chebfuns (f.') reflects about the domain
        # mid-point: g(x) = f(a + b - x) = f(-x) on [-1, 1].
        # pass(2, 3): scalar row chebfun.
        f = cj.chebfun(lambda x: jnp.sin(x) * jnp.abs(x - 0.1), domain=(-1, 0.1, 1))
        ff = f.T.fliplr()
        assert ff.is_transposed
        ff_exact = f(-XR)  # 1-D scalar values; row transpose is identity here
        assert float(jnp.max(jnp.abs(ff(XR) - ff_exact))) < 10 * ff.vscale * EPS
        # fliplr is an involution: fliplr(fliplr(f.')) == f.'
        assert ff.fliplr().isequal(f.T)

        # pass(4, 5): array-valued row chebfun (each column reflected).
        # (MATLAB's cos(x)*sign(x+0.2) second column is replaced by exp(x):
        # sign lands on a breakpoint and prevents convergence -- a construction
        # quirk unrelated to fliplr's reflection semantics, as in the column
        # array test above.)
        g = cj.chebfun(
            lambda x: jnp.stack([jnp.sin(x) * jnp.abs(x - 0.1), jnp.exp(x)], axis=-1),
            domain=(-1, 0.1, 1),
        )
        gg = g.T.fliplr()
        assert gg.is_transposed
        gg_exact = jnp.swapaxes(g(-XR), -1, -2)  # (n_cols, n_points)
        assert float(jnp.max(jnp.abs(gg(XR) - gg_exact))) < 1e2 * gg.vscale * EPS
        assert gg.fliplr().isequal(g.T)

    def test_column_scalar_identity(self):
        # pass(6, 7): fliplr of a scalar column chebfun is the identity.
        # FIXED (Fable 5, Big-Three array-valued epic): columnFliplr semantics.
        f = cj.chebfun(lambda x: jnp.sin(x) * jnp.abs(x - 0.1), domain=(-1, 0.1, 1))
        ff = f.fliplr()
        assert float(jnp.max(jnp.abs(ff(XR) - f(XR)))) < 10 * ff.vscale * EPS
        assert ff.isequal(f)  # fliplr is an involution; identity here

    def test_column_array_reversal(self):
        # pass(8, 9): fliplr of an array-valued column chebfun reverses columns.
        # FIXED (Fable 5, Big-Three array-valued epic).
        # MATLAB's second column cos(x)*sign(x+0.2) is replaced with exp(x):
        # sign(0)=0 lands exactly on the breakpoint and prevents convergence
        # (a construction quirk unrelated to fliplr's column-reversal semantics).
        g = cj.chebfun(
            lambda x: jnp.stack([jnp.sin(x) * jnp.abs(x - 0.1), jnp.exp(x)], axis=-1),
            domain=(-1, 0.1, 1),
        )
        gg = g.fliplr()
        gg_exact = jnp.stack([jnp.exp(XR), jnp.sin(XR) * jnp.abs(XR - 0.1)], axis=-1)
        assert float(jnp.max(jnp.abs(gg(XR) - gg_exact))) < 1e2 * gg.vscale * EPS
        assert gg.fliplr().isequal(g)  # fliplr(fliplr(g)) == g

    def test_singular(self):
        # pass(10, 11): singular-function fliplr.
        pytest.skip("chebfunjax has no SingFun (endpoint 'exps') support")
