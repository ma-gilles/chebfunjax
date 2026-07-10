"""Port of MATLAB Chebfun tests/trigtech/test_conj.m (Opus 4.8).

conj(f) conjugates the Fourier coefficients.

Provenance
----------
MATLAB source : tests/trigtech/test_conj.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechConj:
    @pytest.mark.xfail(reason="chebfunjax trigtech has no conj() method")
    def test_scalar(self):
        raise AssertionError("conj() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no conj() method")
    def test_array(self):
        raise AssertionError("conj() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no conj() method")
    def test_mixed_array(self):
        raise AssertionError("conj() not implemented")

