"""Literal Chebfun2v surface input from MATLAB test_integral2.m.

Provenance
----------
MATLAB source : tests/chebfun3/test_integral2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebpref import ChebfunPref


def test_original_integral2_source_surface_object():
    tol = 1e7 * ChebfunPref().cheb3Prefs.chebfun3eps
    surface = Chebfun2v.from_functions(
        lambda u, v: u * jnp.cos(v),
        lambda u, v: u * jnp.sin(v),
        lambda u, v: v,
        domain=(0, 4, 0, 2 * np.pi),
    )
    f = Chebfun3.from_function(
        lambda x, y, z: jnp.sqrt(1 + x**2 + y**2),
        domain=(-4, 4, -4, 4, 0, 2 * np.pi),
    )
    integral = f.integral2(surface)
    exact = 152 * np.pi / 3
    assert abs(integral - exact) < tol
