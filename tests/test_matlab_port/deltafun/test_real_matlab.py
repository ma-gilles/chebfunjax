"""Port of MATLAB Chebfun tests/deltafun/test_real.m (Opus 4.8).

chebfunjax's Deltafun has no ``real`` method, and delta magnitudes are cast to
real ``float64`` in the constructor (no complex-delta support), so every
assertion in this MATLAB test is skipped with a precise reason.

Provenance
----------
MATLAB source : tests/deltafun/test_real.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="chebfunjax Deltafun has no real() method and delta_mags are cast to "
    "real float64 (no complex-delta support)"
)


class TestDeltafunReal:
    def test_real_empty(self):
        # pass(1): isempty(real(deltafun()))
        pass

    def test_real_of_imag_delta_not_deltafun(self):
        # pass(2): ~isa(real(d), 'deltafun') when deltaMag = 1i
        pass

    def test_real_of_imaginary_delta_is_deltafun(self):
        # pass(3): isa(real(1i*d), 'deltafun')
        pass

    def test_real_magnitude_value(self):
        # pass(4): h.deltaMag == -1 for h = real(1i*d), deltaMag = 1i
        pass

    def test_real_of_complex_matrix(self):
        # pass(5): all(all(real(d).deltaMag == A)) for deltaMag = A + 1i*B
        pass
