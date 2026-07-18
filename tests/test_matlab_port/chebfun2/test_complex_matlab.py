"""Port of MATLAB Chebfun tests/chebfun2/test_complex.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_complex.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="test exercises MATLAB's complex() builtin -- complex(f) and complex(f,f) -- which chebfunjax has no counterpart for; real()/imag()/conj() on Chebfun2 now exist and are tested in test_conj_matlab.py / test_imag_matlab.py")


class TestChebfun2Complex:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
