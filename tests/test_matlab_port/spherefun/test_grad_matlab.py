"""Port of MATLAB Chebfun tests/spherefun/test_grad.m (Fable 5).

Surface gradient of Y_l^m has squared norm integrating to l(l+1)
(Dirichlet energy of an orthonormal harmonic).

Provenance
----------
MATLAB source : tests/spherefun/test_grad.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.spherefun.spherefun import Spherefun

TOL = 1e-7


class TestSpherefunGrad:
    def test_dirichlet_energy_of_harmonic(self):
        l, m = 3, 2
        Y = Spherefun.sphharm(l, m)
        gx, gy, gz = Y.grad()
        e = Spherefun.from_function(
            lambda lam, th: gx(lam, th) ** 2 + gy(lam, th) ** 2
            + gz(lam, th) ** 2)
        assert abs(float(e.sum()) - l * (l + 1)) < 1e3 * TOL
