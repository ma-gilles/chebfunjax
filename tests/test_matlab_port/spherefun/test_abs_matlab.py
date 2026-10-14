"""Source-shaped port of MATLAB tests/spherefun/test_abs.m.

The first test preserves the native continuous infinity norm and current
Chebfun2 preference tolerance.  The sampled check is retained as a separate
supplemental regression control.

Provenance
----------
MATLAB source : tests/spherefun/test_abs.m
Chebfun commit: 7574c77
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun


def _unit_sphere_radius(lam, th):
    x = jnp.cos(lam) * jnp.sin(th)
    y = jnp.sin(lam) * jnp.sin(th)
    z = jnp.cos(th)
    return x**2 + y**2 + z**2


class TestSpherefunAbs:
    def test_original_matlab_continuous_norm_predicate(self):
        tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
        f = Spherefun.from_function(lambda lam, th: -_unit_sphere_radius(lam, th))
        assert float((abs(f) + f).norm("inf")) < tol

    def test_supplemental_sampled_identity_control(self):
        f = Spherefun.from_function(lambda lam, th: -_unit_sphere_radius(lam, th))
        s = abs(f) + f
        lam = jnp.asarray(np.linspace(-np.pi, np.pi, 25))
        th = jnp.asarray(np.linspace(0.0, np.pi, 25))
        LL, TT = jnp.meshgrid(lam, th)
        assert float(jnp.max(jnp.abs(s(LL, TT)))) < 1000 * np.finfo(np.float64).eps
