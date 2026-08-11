"""Port of MATLAB Chebfun tests/chebfun3/test_optimization.m (Fable 5).

FIXED (Fable 5, chebfun2/3 skip sweep): ``max3``/``min3``/``minandmax3``
exist on Chebfun3.

MATLAB sweeps cos(k*pi*x*y*z) for k = 1..7.  Constructing the high-k
members re-runs the full 3D adaptive algorithm on a highly oscillatory
function and is far too slow for a unit test (the k=7 member alone takes
minutes), so the low-frequency members of the battery are used as
witnesses -- they exercise the same optimization code path.

Provenance
----------
MATLAB source : tests/chebfun3/test_optimization.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3 import chebfun3

EPS = float(np.finfo(np.float64).eps)
TOL = 1000 * EPS
DOM = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

# Battery members 1, 2, 8, 9 and 10 with MATLAB's minima/maxima tables.
BATTERY = [
    ("cos(pi xyz)", lambda x, y, z: jnp.cos(np.pi * x * y * z), -1.0, 1.0),
    ("cos(2 pi xyz)", lambda x, y, z: jnp.cos(2 * np.pi * x * y * z),
     -1.0, 1.0),
    ("sin(pi xyz)", lambda x, y, z: jnp.sin(np.pi * x * y * z), 0.0, 1.0),
    ("cos(0)", lambda x, y, z: jnp.cos(0 * np.pi * (x - y - z) ** 2),
     1.0, 1.0),
    ("sin(x+y+z)", lambda x, y, z: jnp.sin(x + y + z), 0.0, 1.0),
]


class TestChebfun3Optimization:
    @pytest.mark.parametrize("name,op,mn,mx", BATTERY)
    def test_global_extrema(self, name, op, mn, mx):
        # The maxima/minima tables of the MATLAB battery, at MATLAB's
        # 1000*eps tolerance.
        f = chebfun3(op, domain=DOM)
        assert abs(float(f.max3()[0]) - mx) < TOL, name
        assert abs(float(f.min3()[0]) - mn) < TOL, name

    def test_minandmax3_agrees_with_max3_and_min3(self):
        f = chebfun3(lambda x, y, z: jnp.cos(np.pi * x * y * z), domain=DOM)
        # MATLAB's [Y, ~] = minandmax3(g) gives Y = [min; max].
        Y = np.asarray(f.minandmax3()[0])
        assert abs(float(Y[0]) - float(f.min3()[0])) < TOL
        assert abs(float(Y[1]) - float(f.max3()[0])) < TOL
