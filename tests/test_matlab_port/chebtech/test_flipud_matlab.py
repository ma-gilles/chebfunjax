"""Port of MATLAB Chebfun tests/chebtech/test_flipud.m (Opus 4.8).

MATLAB ``flipud`` reflects a tech in x (x -> -x).  chebfunjax has NO
``flipud()`` method, so every assertion in this file is skipped with a
precise reason.  No assertion is silently dropped.

Provenance
----------
MATLAB source : tests/chebtech/test_flipud.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

_REASON = "chebfunjax Chebtech has no flipud() method (reflection x -> -x)"


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechFlipud:
    def test_flipud_scalar_real(self, Tech):
        # pass(n,1): flipud(sin(x+.5)) == sin(-x+.5)
        pytest.skip(_REASON)

    def test_flipud_array_real(self, Tech):
        # pass(n,2): flipud([sin(x+.5), exp(x)]) == [sin(-x+.5), exp(-x)]
        pytest.skip(_REASON)

    def test_flipud_scalar_complex(self, Tech):
        # pass(n,3): flipud(sin(1i*x+.5)) == sin(-1i*x+.5)
        pytest.skip(_REASON)

    def test_flipud_array_complex(self, Tech):
        # pass(n,4): flipud([sin(x+.5), exp(1i*x)]) == [sin(-x+.5), exp(-1i*x)]
        pytest.skip(_REASON)
