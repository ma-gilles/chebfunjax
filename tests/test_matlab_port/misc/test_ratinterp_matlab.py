"""Port of MATLAB Chebfun tests/misc/test_ratinterp.m (Fable 5).

The type0 (roots-of-unity) case is ported; type1/type2 grid variants
are xfailed (chebfunjax ratinterp exposes one grid type).

Provenance
----------
MATLAB source : tests/misc/test_ratinterp.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
import pytest

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

    @pytest.mark.xfail(reason="chebfunjax ratinterp has no type1/type2 "
                       "(Chebyshev-grid) variants")
    def test_grid_type_variants(self):
        ratinterp(f, 10, 10, domain=(0.0, 2.0), grid="type1")
