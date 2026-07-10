"""Port of MATLAB Chebfun tests/chebtech/test_sample.m (Opus 4.8).

MATLAB ``[v, p] = sample(f)`` / ``sample(f, m)`` returns the values ``v`` of the
chebtech on an ``m``-point Chebyshev grid together with the grid ``p``.
chebfunjax has NO ``sample`` method; the equivalent is
``p = chebpts(m, kind); v = f(p)`` (feval on the grid), but there is no
``sample`` API to exercise, so the whole test is skipped.

Provenance
----------
MATLAB source : tests/chebtech/test_sample.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

CASES = [(Chebtech1, 1), (Chebtech2, 2)]

_NO_SAMPLE = "chebfunjax has no chebtech.sample; chebpts+feval is the equivalent"


class TestChebtechSample:
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_grid_equal_length(self, Tech, kind):
        # pass(n, 1): sample(f) on a grid equal to length(f).
        pytest.skip(_NO_SAMPLE)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_grid_shorter(self, Tech, kind):
        # pass(n, 2): sample(f, m) with m < length(f).
        pytest.skip(_NO_SAMPLE)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_grid_longer(self, Tech, kind):
        # pass(n, 3): sample(f, m) with m > length(f).
        pytest.skip(_NO_SAMPLE)
