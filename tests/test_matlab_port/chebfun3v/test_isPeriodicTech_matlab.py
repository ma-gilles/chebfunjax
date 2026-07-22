"""Port of MATLAB Chebfun tests/chebfun3v/test_isPeriodicTech.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_isPeriodicTech.m
Chebfun commit: 7574c77

Notes
-----
The MATLAB test's assertions 2-4 build 'trig' CHEBFUN3s and expect
isPeriodicTech to report True.  chebfunjax Chebfun3 is always built on a
Chebyshev tech (there is no trigonometric tech), so isPeriodicTech is always
False and only the non-periodic assertion (pass 1) is reachable.
"""

from __future__ import annotations

from chebfunjax.chebfun3d.chebfun3v import Chebfun3v


class TestChebfun3vIsperiodictech:
    def test_non_periodic_field(self):
        f = Chebfun3v.from_functions(lambda x, y, z: x, lambda x, y, z: y)
        assert not f.isPeriodicTech()
