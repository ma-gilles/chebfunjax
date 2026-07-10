"""Port of MATLAB Chebfun tests/chebtech/test_sign.m (Opus 4.8).

The MATLAB file loops ``for type = 1:2`` over ``{chebtech1(), chebtech2()}``
and checks ``sign(f)`` for positive/negative/complex/complex-array functions.
chebfunjax implements NO ``sign`` on either Chebtech1 or Chebtech2, so every
assertion is an honest xfail.

Provenance
----------
MATLAB source : tests/chebtech/test_sign.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

_REASON = "chebfunjax lacks sign()"


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechSign:
    def test_sign_positive(self, Tech):
        # pass(type, 1): sign(sin(x) + 2) == 1.
        pytest.xfail(_REASON)

    def test_sign_negative(self, Tech):
        # pass(type, 2): sign(-(sin(x) + 2)) == -1.
        pytest.xfail(_REASON)

    def test_sign_complex(self, Tech):
        # pass(type, 3): sign(exp(1i pi x)) == exp(1i pi x).
        pytest.xfail(_REASON)

    def test_sign_complex_array(self, Tech):
        # pass(type, 4): complex array-valued sign.
        pytest.xfail(_REASON)
