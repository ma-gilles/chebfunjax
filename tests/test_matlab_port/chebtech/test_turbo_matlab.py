"""Port of MATLAB Chebfun tests/chebtech/test_turbo.m (Opus 4.8).

MATLAB's "turbo" construction (``pref.useTurbo = true``) computes twice as many
Chebyshev coefficients using a convergence-acceleration technique.  chebfunjax
has no ``useTurbo`` preference / turbo construction option, so every assertion
is skipped.

Provenance
----------
MATLAB source : tests/chebtech/test_turbo.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

CASES = [(Chebtech1, 1), (Chebtech2, 2)]

_NO_TURBO = "chebfunjax has no useTurbo pref / turbo construction option"
# Array-valued techs are now supported; the sole remaining blocker is turbo.
_NO_TURBO_ARRAY = _NO_TURBO


class TestChebtechTurbo:
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_length_doubles(self, Tech, kind):
        # pass(n, 1): length(f_turbo) == 2*length(f_plain).
        pytest.skip(_NO_TURBO)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_isreal(self, Tech, kind):
        # pass(n, 2): isreal(f_turbo).
        pytest.skip(_NO_TURBO)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_coeff_accuracy(self, Tech, kind):
        # pass(n, 3): coeffs match 2*besseli(k,1) to tol.
        pytest.skip(_NO_TURBO)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_array_length(self, Tech, kind):
        # pass(n, 4)
        pytest.skip(_NO_TURBO_ARRAY)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_array_isreal(self, Tech, kind):
        # pass(n, 5)
        pytest.skip(_NO_TURBO_ARRAY)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_array_coeff_accuracy(self, Tech, kind):
        # pass(n, 6)
        pytest.skip(_NO_TURBO_ARRAY)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_fixedlength(self, Tech, kind):
        # pass(n, 7): length(f_turbo) == 75 with fixedLength.
        pytest.skip(_NO_TURBO)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_fixedlength_accuracy(self, Tech, kind):
        # pass(n, 8)
        pytest.skip(_NO_TURBO)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_fixedlength_array(self, Tech, kind):
        # pass(n, 9)
        pytest.skip(_NO_TURBO_ARRAY)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_fixedlength_array_accuracy(self, Tech, kind):
        # pass(n, 10)
        pytest.skip(_NO_TURBO_ARRAY)
