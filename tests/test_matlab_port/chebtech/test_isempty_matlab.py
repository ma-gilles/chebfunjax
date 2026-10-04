"""Port of MATLAB Chebfun tests/chebtech/test_isempty.m (Fable 5).

chebfunjax techs have a genuine empty representation (``Tech.empty()`` /
``isempty()``) and genuine array-valued (n, m) coefficient matrices, so
pass(n, 1)-(4) port directly.  MATLAB's horizontal concatenation
``[f g]`` of techs maps to ``Tech.cell2mat([f, g])``.

All five source groups execute; horizontal concatenation discards empty
arguments and returns the first object unchanged when all are empty.

Provenance
----------
MATLAB source : tests/chebtech/test_isempty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechIsempty:
    def test_empty_tech_is_empty(self, Tech):
        # pass(n,1): isempty(make())
        assert Tech.empty().isempty()

    def test_constructed_tech_is_not_empty(self, Tech):
        # pass(n,2): ~isempty(make(@sin))
        f = Tech.from_function(jnp.sin)
        assert not f.isempty()
        assert f.n != 0

    def test_array_valued_not_empty(self, Tech):
        # pass(n,3): ~isempty(make(@(x) [sin(x), cos(x)]))
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        assert f.coeffs.ndim == 2 and f.coeffs.shape[1] == 2
        assert not f.isempty()

    def test_horzcat_not_empty(self, Tech):
        # pass(n,4): ~isempty([make(@sin), make(@sin)])
        f = Tech.from_function(jnp.sin)
        g = Tech.cell2mat([f, f])
        assert g.coeffs.shape[1] == 2
        assert not g.isempty()

    def test_horzcat_empty_is_empty(self, Tech):
        # pass(n,5): isempty([make(), make()])
        first = Tech.empty()
        second = Tech.empty()
        result = Tech.cell2mat([first, second])
        assert result.isempty()
        assert result is first
