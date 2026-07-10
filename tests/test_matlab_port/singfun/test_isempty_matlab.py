"""Port of MATLAB Chebfun tests/singfun/test_isempty.m (Opus 4.8).

MATLAB distinguishes an empty singfun (``singfun()``), a zero singfun
(``singfun.zeroSingFun()``), and a non-empty singfun.  chebfunjax has neither
an empty Singfun representation, a ``zeroSingFun`` factory, nor an ``isempty``
method (a Singfun always wraps a non-empty smooth part), so every assertion is
skipped.

Provenance
----------
MATLAB source : tests/singfun/test_isempty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestSingfunIsempty:
    def test_empty_is_empty(self):
        pytest.skip("chebfunjax has no empty Singfun representation / isempty")

    def test_zerosingfun_not_empty(self):
        pytest.skip("chebfunjax has no zeroSingFun factory / isempty")

    def test_nonzero_not_empty(self):
        pytest.skip("chebfunjax has no isempty method (Singfun is never empty)")
