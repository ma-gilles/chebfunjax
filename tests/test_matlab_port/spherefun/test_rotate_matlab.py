"""Port of MATLAB Chebfun tests/spherefun/test_rotate.m (Fable 5).

Original deterministic clauses4–8 use source operands and continuous norms.
The retained analytic roundtrip is supplemental; original random clauses1–2
and the feval/nufft method comparison in clause3 remain unqualified.

Provenance
----------
MATLAB source : tests/spherefun/test_rotate.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun

TOL = 1e3 * np.finfo(float).eps
ANGLS = (0.704147450733711, 1.158707918472494, 0.884902035498222)

LAMS = jnp.asarray(np.linspace(-3.0, 3.0, 13))
THS = jnp.asarray(np.linspace(0.05, 3.09, 13))
LL, TT = jnp.meshgrid(LAMS, THS, indexing="ij")


def _maxdiff(f, g):
    return float(jnp.max(jnp.abs(f(LL, TT) - g(LL, TT))))


def _source_tol():
    return 1e3 * float(ChebfunPref().cheb2Prefs.chebfun2eps)


class TestSpherefunRotate:
    # (This carried a non-strict xfail for a 2026-07 CI measurement of
    # 2.36e-12 vs the 2.2e-12 bound; it has passed 4 consecutive CI runs
    # and locally at 9.6e-13 -- mark removed 2026-07-30, restoring strict
    # enforcement at MATLAB's 10*TOL roundtrip bound.)
    def test_rotate_and_undo(self):
        # Supplemental analytic roundtrip on a finite evaluation grid.
        # This is not the original feval/nufft comparison in pass(3).
        f = Spherefun.from_function(
            lambda lam, th: jnp.sin(
                jnp.cos(th) + jnp.cos(lam - 0.2) * jnp.sin(th)
                + jnp.sin(lam + 0.4) * jnp.sin(th)) ** 8)
        g = f.rotate(*ANGLS)
        h = g.rotate(-ANGLS[2], -ANGLS[1], -ANGLS[0])
        assert _maxdiff(h, f) < 10 * TOL

    def test_original_clause4_integral_preserved(self):
        # Original deterministic pass(4): integral preserved under rotation.
        f = Spherefun.from_function(
            lambda lam, th: jnp.sin(
                jnp.cos(th) + jnp.cos(lam - 0.2) * jnp.sin(th)
                + jnp.sin(lam + 0.4) * jnp.sin(th)) ** 8)
        g = f.rotate(*ANGLS)
        assert abs(float(g.sum2()) - float(f.sum2())) < _source_tol()

    def test_original_clause5_sphharm21_m0_invariant_under_z_rotation(self):
        # Original deterministic pass(5): sphharm(21,0), unchanged by Rz.
        f = Spherefun.sphharm(21, 0)
        g = f.rotate(np.sqrt(2.0), 0.0, 0.0)
        assert float((f - g).norm()) < _source_tol()

    def test_original_clause6_m10_is_invariant_under_source_z_rotation(self):
        # MATLAB tests/spherefun/test_rotate.m pass(6), same operand, angle,
        # continuous sphere norm, and preference-derived tolerance.
        f = Spherefun.sphharm(10, 10)
        g = f.rotate(np.pi / 5.0, 0.0, 0.0)
        assert float((f - g).norm()) < _source_tol()

    def test_original_clauses7_and8_octahedral_symmetry(self):
        # MATLAB tests/spherefun/test_rotate.m passes(7:8), preserving the
        # exact harmonics, coefficient, angles, norm, and tolerance.
        f = (Spherefun.sphharm(4, 0)
             + np.sqrt(5.0 / 7.0) * Spherefun.sphharm(4, 4))
        g = f.rotate(np.pi / 2.0, np.pi / 2.0, 0.0)
        assert float((f - g).norm()) < _source_tol()
        g = g.rotate(np.pi / 2.0, np.pi / 2.0, 0.0)
        assert float((f - g).norm()) < _source_tol()
