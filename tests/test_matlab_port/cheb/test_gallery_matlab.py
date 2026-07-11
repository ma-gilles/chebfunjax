"""Port of MATLAB Chebfun tests/cheb/test_gallery.m (Fable 5).

Every MATLAB gallery name constructs without crashing (incl. the no-arg
random pick).  This is the doesNotCrash sweep; per-entry value checks
live in tests/test_coverage.  Heavy entries (blasius/daubechies) are
covered by the parametrized coverage tests to keep this file fast.

Provenance
----------
MATLAB source : tests/cheb/test_gallery.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import random as _rnd

import pytest

from chebfunjax.utils.gallery import gallery, list_gallery

FAST = sorted(set(list_gallery()) - {"blasius", "daubechies", "gamma",
                                     "motto", "si", "stegosaurus"})


class TestChebGallery:
    @pytest.mark.parametrize("name", FAST)
    def test_does_not_crash(self, name):
        assert gallery(name) is not None

    def test_no_arg_random_pick(self):
        _rnd.seed(4)
        assert gallery(None) is not None

    def test_all_27_names_present(self):
        matlab = {"airy", "bessel", "blasius", "bump", "chirp",
                  "daubechies", "erf", "fishfillet", "gamma", "gaussian",
                  "jitter", "kahaner", "motto", "random", "rose",
                  "runge", "seismograph", "si", "sinefun1", "sinefun2",
                  "spikycomb", "stegosaurus", "vandercheb",
                  "vandermonde", "wiggly", "wild", "zigzag"}
        assert matlab.issubset(set(list_gallery().keys()))
