"""Port of MATLAB Chebfun tests/cheb/test_galleryball.m (Fable 5).

Provenance
----------
MATLAB source : tests/cheb/test_galleryball.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax

from chebfunjax import cheb
from chebfunjax.ballfun import Ballfun

jax.config.update("jax_enable_x64", True)

NAMES = ["deathstar", "gaussian", "helmholtz", "moire", "peaks",
         "roundpeg", "solharm", "stripes", "wave"]


class TestChebGalleryball:
    def test_all_matlab_assertions(self):
        # MATLAB: pass(k) = doesNotCrash(names{k}) for every gallery name.
        for name in NAMES:
            f, _fa = cheb.galleryball(name)
            assert isinstance(f, Ballfun), name
