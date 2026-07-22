"""Port of MATLAB Chebfun tests/chebop2/test_linearSchrodinger.m (Fable 5).

In the reference (commit 7574c77) the entire body of this test is commented
out and the function simply returns ``pass = 1`` -- the linear Schrodinger
example was disabled upstream.  The faithful port therefore asserts the same
trivially-true result.

Provenance
----------
MATLAB source : tests/chebop2/test_linearSchrodinger.m
Chebfun commit: 7574c77
"""

from __future__ import annotations


class TestChebop2Linearschrodinger:
    def test_disabled_upstream_passes(self):
        # MATLAB body is fully commented out; test returns pass = 1.
        assert True
