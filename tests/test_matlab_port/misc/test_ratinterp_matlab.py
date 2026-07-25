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

from chebfunjax.utils.ratapprox import ratinterp

TOL = 1e-10


def f(x):
    return (x ** 4 - 3) / ((x + 0.2) * (x - 2.2))


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
