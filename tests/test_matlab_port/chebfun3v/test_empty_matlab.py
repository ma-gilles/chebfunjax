"""Port of MATLAB Chebfun tests/chebfun3v/test_empty.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_empty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.chebfun3d.chebfun3v import Chebfun3v


class TestChebfun3vEmpty:
    def test_empty_operations_do_not_crash(self):
        # Mirror of MATLAB test_empty: every command must accept the empty
        # CHEBFUN3V without error.
        F = Chebfun3v()
        F.conj()
        F.cross(F)
        F.transpose()
        F.curl()
        F.divergence()
        F.dot(F)
        F(1, 1, 1)
        F.imag()
        assert F.isempty()
        F.laplacian()
        F - F
        F * F
        F.norm()
        F + F
        F ** 2
        F.real()
        F.roots()
        F.ctranspose()
