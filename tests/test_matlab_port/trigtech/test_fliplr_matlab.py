"""Port of MATLAB Chebfun tests/trigtech/test_fliplr.m (Opus 4.8).

fliplr reverses the column order of an array-valued trigtech.

Provenance
----------
MATLAB source : tests/trigtech/test_fliplr.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechFliplr:
    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / fliplr")
    def test_scalar_identity(self):
        raise AssertionError("fliplr not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / fliplr")
    def test_swap_columns(self):
        raise AssertionError("fliplr not implemented")

