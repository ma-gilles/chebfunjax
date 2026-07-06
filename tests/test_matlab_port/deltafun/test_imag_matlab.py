"""Port of MATLAB Chebfun tests/deltafun/test_imag.m (Opus 4.8).

chebfunjax's Deltafun has no ``imag`` method, and delta magnitudes are cast to
real ``float64`` in the constructor (no complex-delta support), so every
assertion in this MATLAB test is skipped with a precise reason.

Provenance
----------
MATLAB source : tests/deltafun/test_imag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="chebfunjax Deltafun has no imag() method and delta_mags are cast to "
    "real float64 (no complex-delta support)"
)


class TestDeltafunImag:
    def test_imag_empty(self):
        # pass(1): isempty(imag(deltafun()))
        pass

    def test_imag_of_real_delta_not_deltafun(self):
        # pass(2): ~isa(imag(d), 'deltafun') when deltaMag is real
        pass

    def test_imag_of_imaginary_delta_is_deltafun(self):
        # pass(3): isa(imag(1i*d), 'deltafun')
        pass

    def test_imag_magnitude_value(self):
        # pass(4): h.deltaMag == 1 for h = imag(1i*d)
        pass

    def test_imag_of_complex_matrix(self):
        # pass(5): all(all(imag(d).deltaMag == B)) for deltaMag = A + 1i*B
        pass
