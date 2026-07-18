"""Port of MATLAB Chebfun tests/chebfun/test_any.m (Fable 5).

``any(f)`` is True where the function is nonzero somewhere on its domain.
For array-valued chebfuns it returns a per-column ``(m,)`` boolean array
(``any`` down the continuous dimension).

The ``any(f, 2)`` cases (a chebfun-valued reduction across the discrete
dimension), the row-chebfun / transpose cases, and the pointValues cases
have no chebfunjax counterpart and stay skipped.

Provenance
----------
MATLAB source : tests/chebfun/test_any.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj


class TestChebfunAny:
    def test_empty(self):
        # pass(1): ~any(chebfun()).
        pytest.skip("chebfunjax has no empty chebfun")

    def test_columns(self):
        # pass(2): any([sin(x) 0*x exp(x)]) == [1 0 1].
        # FIXED (Fable 5, Big-Three array-valued epic): per-column any() -> (m,).
        f = cj.chebfun(
            lambda x: jnp.stack([jnp.sin(x), 0 * x, jnp.exp(x)], axis=-1),
            domain=(-1, -0.5, 0, 0.5, 1),
        )
        assert list(np.asarray(f.any()).astype(int)) == [1, 0, 1]

    def test_columns_with_complex(self):
        # pass(5): any([0*x hvsde(x) exp(2 pi i x)]) == [0 1 1].
        # FIXED (Fable 5, Big-Three array-valued epic).
        hvsde = lambda x: 0.5 * (jnp.sign(x) + 1)
        f = cj.chebfun(
            lambda x: jnp.stack([0 * x, hvsde(x), jnp.exp(2 * np.pi * 1j * x)], axis=-1),
            domain=(-1, 0, 1),
        )
        assert list(np.asarray(f.any()).astype(int)) == [0, 1, 1]

    def test_transpose_rows(self):
        # pass(3, 6): any(f.', 2) on a row chebfun.
        pytest.skip("chebfunjax has no row-chebfun transpose")

    def test_pointvalues_nan(self):
        # pass(4): any(f, 1) after setting a pointValues entry to NaN.
        pytest.skip("chebfunjax Chebfun has no pointValues field")

    def test_discrete_dimension(self):
        # pass(7-11): any(f, 2) reduces across columns to a chebfun.
        pytest.skip(
            "Chebfun.any() has no dim-2 mode (a chebfun-valued reduction across "
            "the discrete dimension); it reduces to a per-column boolean"
        )

    def test_dim_error(self):
        # pass(12): any(f, 3) raises CHEBFUN:CHEBFUN:any:dim.
        pytest.skip("Chebfun.any() takes no dim argument")

    def test_singular_and_unbounded(self):
        # pass(13-16): singular / unbounded-domain cases.
        pytest.skip("chebfunjax has no SingFun or unbounded-domain support")
