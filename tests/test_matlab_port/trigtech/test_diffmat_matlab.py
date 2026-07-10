"""Port of MATLAB Chebfun tests/trigtech/test_diffmat.m (Opus 4.8).

diffmat builds the trigcolloc spectral differentiation matrix.

Provenance
----------
MATLAB source : tests/trigtech/test_diffmat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechDiffmat:
    @pytest.mark.xfail(reason="chebfunjax lacks trigcolloc / diffmat (differentiation via diff() only)")
    def test_d1_odd(self):
        raise AssertionError("diffmat not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks trigcolloc / diffmat (differentiation via diff() only)")
    def test_d1_odd_hard(self):
        raise AssertionError("diffmat not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks trigcolloc / diffmat (differentiation via diff() only)")
    def test_d1_even(self):
        raise AssertionError("diffmat not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks trigcolloc / diffmat (differentiation via diff() only)")
    def test_d1_even_complex(self):
        raise AssertionError("diffmat not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks trigcolloc / diffmat (differentiation via diff() only)")
    def test_d2_odd(self):
        raise AssertionError("diffmat not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks trigcolloc / diffmat (differentiation via diff() only)")
    def test_d2_even(self):
        raise AssertionError("diffmat not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks trigcolloc / diffmat (differentiation via diff() only)")
    def test_d5_odd(self):
        raise AssertionError("diffmat not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks trigcolloc / diffmat (differentiation via diff() only)")
    def test_d5_even(self):
        raise AssertionError("diffmat not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks trigcolloc / diffmat (differentiation via diff() only)")
    def test_d6_odd(self):
        raise AssertionError("diffmat not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks trigcolloc / diffmat (differentiation via diff() only)")
    def test_d6_even(self):
        raise AssertionError("diffmat not implemented")

