"""Port of MATLAB Chebfun tests/chebfun/test_permute.m (Fable 5).

``permute(f, [1 2])`` is the identity; ``permute(f, [2 1])`` is the
row/column transpose ``f.'``.  Any other order errors.  Row-chebfun
orientation was added with the transpose feature (Fable 5), so this test is
now a direct port.

Provenance
----------
MATLAB source : tests/chebfun/test_permute.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunPermute:
    def test_identity_permute(self):
        # pass(1): norm(f - permute(f, [1 2])) == 0.
        f = cj.chebfun(lambda x: jnp.sin(x))
        g = f.permute([1, 2])
        assert not g.is_transposed
        assert float((f - g).norm()) == 0.0

    def test_transpose_permute(self):
        # pass(2): norm(f.' - permute(f, [2 1])) == 0.
        f = cj.chebfun(lambda x: jnp.sin(x))
        g = f.permute([2, 1])
        assert g.is_transposed
        assert g.isequal(f.T)
        # size flips from (inf, 1) to (1, inf).
        assert g.size() == (1, float("inf"))

    def test_double_permute_is_identity(self):
        f = cj.chebfun(lambda x: jnp.cos(3 * x) + x)
        g = f.permute([2, 1]).permute([2, 1])
        assert not g.is_transposed
        assert g.isequal(f)

    @pytest.mark.parametrize("order", [[1, 1], [1, 3], [2, 2]])
    def test_invalid_order_errors(self, order):
        # pass(3, 4): permute(f, [1 1]) / [1 3] must raise.
        f = cj.chebfun(lambda x: jnp.sin(x))
        with pytest.raises(ValueError):
            f.permute(order)
