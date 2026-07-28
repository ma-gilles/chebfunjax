"""Port of MATLAB Chebfun tests/misc/test_ratinterp.m (Fable 5).

Ports the type0 (roots-of-unity) case together with the type1 (1st-kind
Chebyshev) and type2 (2nd-kind Chebyshev) grid variants; all recover the
same type-(4, 2) approximant with poles at -0.2 and 2.2.

Provenance
----------
MATLAB source : tests/misc/test_ratinterp.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.utils.quadrature import chebpts
from chebfunjax.utils.ratapprox import ratinterp

TOL = 1e-10


def f(x):
    return (x ** 4 - 3) / ((x + 0.2) * (x - 2.2))


def _mapab(x, a, b):
    return a + (b - a) * (x + 1) / 2


class TestRatinterp:
    def test_degrees_and_poles(self):
        p, q, r, mu, nu, poles, res = ratinterp(f, 10, 10,
                                                domain=(0.0, 2.0))
        assert mu == 4 and nu == 2
        pl = np.sort_complex(np.asarray(poles))
        assert float(np.max(np.abs(pl - np.array([-0.2, 2.2])))) < TOL

    def test_grid_type_variants(self):
        # MATLAB pass(2)/pass(3): the type1 (1st-kind Chebyshev) and type2
        # (2nd-kind Chebyshev) grids both yield the type-(4, 2) approximant
        # with poles at -0.2 and 2.2.
        for grid in ("type1", "type2"):
            p, q, r, mu, nu, poles, res = ratinterp(f, 10, 10,
                                                    domain=(0.0, 2.0), xi=grid)
            assert mu == 4 and nu == 2
            pl = np.sort_complex(np.asarray(poles))
            assert float(np.max(np.abs(pl - np.array([-0.2, 2.2])))) < TOL

    def test_data_vector_reduction(self):
        # MATLAB pass(9)/(10)/(11): a length-N=100 vector of samples on
        # 1st-/2nd-kind Chebyshev and equispaced grids still reduces the
        # requested type-(10, 10) to the exact type-(4, 2).
        N = 100
        x1 = _mapab(np.asarray(chebpts(N, kind=1)), 0, 2)
        _, _, _, mu, nu, _, _ = ratinterp(f(x1), 10, 10, N, "type1",
                                          domain=(0.0, 2.0))
        assert mu == 4 and nu == 2
        x2 = _mapab(np.asarray(chebpts(N, kind=2)), 0, 2)
        _, _, _, mu, nu, _, _ = ratinterp(f(x2), 10, 10, N, "type2",
                                          domain=(0.0, 2.0))
        assert mu == 4 and nu == 2
        xe = np.linspace(0, 2, N)
        _, _, _, mu, nu, _, _ = ratinterp(f(xe), 10, 10, N, "equi",
                                          domain=(0.0, 2.0))
        assert mu == 4 and nu == 2

    def test_simple_pole(self):
        # MATLAB pass(12): 1/(x - 0.2) requested (10, 10) reduces to type
        # (0, 1) with the single pole at 0.2.
        _, _, _, mu, nu, poles, _ = ratinterp(lambda x: 1.0 / (x - 0.2),
                                              10, 10, xi="type2")
        assert mu == 0 and nu == 1
        assert abs(float(np.real(poles[0])) - 0.2) < 1e-10
