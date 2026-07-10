"""Port of MATLAB Chebfun tests/chebtech/test_isempty.m (Opus 4.8).

chebfunjax Chebtech has no ``isempty()`` method, but an empty tech built
from an empty coefficient vector has ``n == 0`` and a constructed
non-empty tech has ``n != 0`` — the faithful scalar equivalent of the
MATLAB predicate.  The array-valued / quasimatrix horzcat cases have no
scalar-valued analogue and are skipped.

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
        # pass(n,1): isempty(make())  ->  empty coeffs give n == 0
        f = Tech.from_coeffs(jnp.array([]))
        assert f.n == 0

    def test_constructed_tech_is_not_empty(self, Tech):
        # pass(n,2): ~isempty(make(@sin))  ->  non-empty tech has n != 0
        f = Tech.from_function(jnp.sin)
        assert f.n != 0

    def test_array_valued_not_empty(self, Tech):
        # pass(n,3): ~isempty(make(@(x) [sin(x), cos(x)]))
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix techs"
        )

    def test_horzcat_not_empty(self, Tech):
        # pass(n,4): ~isempty([make(@sin), make(@sin)])
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix techs"
        )

    def test_horzcat_empty_is_empty(self, Tech):
        # pass(n,5): isempty([make(), make()])
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix techs"
        )
