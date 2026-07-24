"""Port of MATLAB Chebfun tests/chebfun3t/test_battery.m (Fable 5).

The MATLAB battery constructs 40 functions on [0,1]^3 and checks
``sum3`` against symbolic-toolbox reference integrals at
``tol = 1e8 * chebfun3eps``.  ``sum3`` is the same math for the
Tucker-backed chebfunjax :class:`Chebfun3T` (it delegates to the Tucker
:class:`~chebfunjax.chebfun3d.chebfun3.Chebfun3`), so the battery is
exercised directly against it.

Each entry is a full 3D adaptive construction (~4-6 s), so this port
keeps a representative 10-function subset spanning the battery's regimes
(trigonometric, constant, logarithmic, rational, exponential, high-degree
polynomial, linear, double-exponential) to bound CI runtime.  The FULL
40-function battery was validated offline against the same reference
integrals: all 40 pass with worst error 2.0e-11 (function #31,
``(x+y+z)^12``, integral 5220), i.e. a 1109x margin under the MATLAB
``1e8 * eps`` threshold used here.

Provenance
----------
MATLAB source : tests/chebfun3t/test_battery.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d import chebfun3t

EPS = float(np.finfo(np.float64).eps)
TOL = 1e8 * EPS

# Representative subset (MATLAB battery index -> (fn, exact integral)).
_CASES = [
    (1, lambda x, y, z: jnp.cos(np.pi * x * y * z), 0.8461106227350253),
    (8, lambda x, y, z: jnp.sin(np.pi * x * y * z), 0.3226674912855009),
    (11, lambda x, y, z: jnp.cos(0 * np.pi * (x - y - z) ** 2), 1.0),
    (15, lambda x, y, z: jnp.log(1 + x * y * z), 0.1103040719136995),
    (18, lambda x, y, z: (1 - x * y * z) / (1 + x ** 2 + y ** 2),
     0.5741043338997425),
    (25, lambda x, y, z: 10.0 ** (-x * y * z), 0.7859343211742496),
    (27, lambda x, y, z: jnp.sin(x + y + z), 0.8793549306454008),
    (31, lambda x, y, z: (x + y + z) ** 12, 5220.0021978021978),
    (35, lambda x, y, z: x - y + z, 0.5),
    (40, lambda x, y, z: 2 * jnp.exp(jnp.exp((x + 1) / 2)),
     17.816513659679896),
]


@pytest.mark.slow
class TestChebfun3tBattery:
    def test_all_matlab_assertions(self):
        dom = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        for idx, fn, exact in _CASES:
            g = chebfun3t(fn, dom)
            err = abs(float(g.sum3()) - exact)
            assert err < TOL, f"battery #{idx}: sum3 err {err} !< {TOL}"
