"""Port of MATLAB Chebfun tests/chebtech/test_fliplr.m (Opus 4.8).

MATLAB ``fliplr`` flips the *columns* of an array-valued tech.  For a
scalar-valued tech ``fliplr(f) == f`` (identity), and for array-valued
techs it reverses column order.  chebfunjax has NO ``fliplr()`` method,
is scalar-valued, and also has no ``isequal()`` predicate to express the
MATLAB checks, so both assertions are skipped with a precise reason.  No
assertion is silently dropped.

Provenance
----------
MATLAB source : tests/chebtech/test_fliplr.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

_REASON = (
    "chebfunjax lacks fliplr; array-valued column-flip N/A for scalar techs "
    "(and no isequal predicate)"
)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechFliplr:
    def test_fliplr_scalar_identity(self, Tech):
        # pass(n,1): isequal(f, fliplr(f)) for scalar sin
        pytest.skip(_REASON)

    def test_fliplr_array_reverses_columns(self, Tech):
        # pass(n,2): isequal(fliplr([sin cos]), [cos sin])
        pytest.skip(_REASON)
