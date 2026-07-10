"""Port of MATLAB Chebfun tests/chebtech/test_chebTcoeffs2chebUcoeffs.m (Opus 4.8).

MATLAB ``chebtech.chebTcoeffs2chebUcoeffs`` converts first-kind Chebyshev (T)
coefficients to second-kind Chebyshev (U) coefficients.  A source grep confirms
chebfunjax has NO such function anywhere, so every assertion is xfail'd.

Provenance
----------
MATLAB source : tests/chebtech/test_chebTcoeffs2chebUcoeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech2

_NO_CONV = "chebfunjax lacks chebTcoeffs2chebUcoeffs (T->U coefficient conversion)"


class TestChebTcoeffs2chebUcoeffs:
    @pytest.mark.xfail(reason=_NO_CONV, strict=False)
    def test_empty(self):
        # pass(1): isempty(chebTcoeffs2chebUcoeffs([])).
        Chebtech2.chebTcoeffs2chebUcoeffs(jnp.array([]))

    @pytest.mark.xfail(reason=_NO_CONV, strict=False)
    def test_column_vector(self):
        # pass(2): cT -> cU for 1 + x + x^2 + x^3 + x^4.
        cT = jnp.array([1.875, 1.75, 1.0, 0.25, 0.125])
        Chebtech2.chebTcoeffs2chebUcoeffs(cT)

    @pytest.mark.xfail(reason=_NO_CONV, strict=False)
    def test_matrix(self):
        # pass(3): eye(5) -> recurrence matrix.
        Chebtech2.chebTcoeffs2chebUcoeffs(jnp.eye(5))
