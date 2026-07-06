"""Port of MATLAB Chebfun tests/deltafun/test_isempty.m (Opus 4.8).

chebfunjax has no empty Deltafun (a ``funPart`` is always required) and no
``isempty`` method, so every assertion in this MATLAB test is skipped with a
precise reason.

Provenance
----------
MATLAB source : tests/deltafun/test_isempty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="chebfunjax has no empty Deltafun and no isempty() method "
    "(a funPart is always required)"
)


class TestDeltafunIsempty:
    def test_empty_constructor(self):
        # pass(1): isempty(deltafun())
        pass

    def test_empty_bndfun_funpart(self):
        # pass(2): isempty(deltafun(bndfun([])))
        pass

    def test_empty_delta_arg(self):
        # pass(3): isempty(deltafun(f, []))
        pass

    def test_empty_delta_struct(self):
        # pass(4): isempty(deltafun(f, struct('deltaMag', [], 'deltaLoc', [])))
        pass
