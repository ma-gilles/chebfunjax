"""Core-suite tests for the vectorised spherical-harmonic helpers and the
Spherefun near-zero arithmetic short-circuit (Fable 5).

These back the fix for the nested-composition XLA CPU compile blow-up:

* ``_all_real_ylm_values`` / ``_sph_harmonic_eval_sum`` share the
  associated-Legendre recurrence across ``(l, m)`` instead of restarting it
  per harmonic, so they must stay bit-identical to the per-``(l, m)``
  reference ``_real_ylm_values``.
* ``Spherefun._binary`` returns the exact zero field when a result is only
  rounding noise relative to the operands, which is what makes the surface
  vector-calculus identities (``div(grad f) == laplacian f`` etc.) both
  finish quickly and hold exactly.

Provenance
----------
Mirrors of MATLAB @spherefun behaviour; Chebfun commit 7574c77.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import (
    Spherefun,
    _all_real_ylm_values,
    _real_ylm_values,
    _sph_harmonic_eval_sum,
)

LAM = jnp.asarray(np.linspace(-3.0, 3.0, 13))
TH = jnp.asarray(np.linspace(0.15, 2.95, 13))
LL, TT = jnp.meshgrid(LAM, TH, indexing="ij")


def _snorm(sf) -> float:
    return float(jnp.max(jnp.abs(np.asarray(sf(LL, TT)))))


class TestVectorisedHarmonics:
    def test_all_ylm_matches_reference(self):
        # Shared-recurrence evaluator is bit-identical to the per-(l, m) one.
        lmax = 10
        lam = jnp.asarray(np.linspace(-np.pi, np.pi, 40))
        th = jnp.asarray(np.linspace(0.05, np.pi - 0.05, 40))
        Y = _all_real_ylm_values(lmax, lam, th)
        max_err = 0.0
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                ref = np.asarray(_real_ylm_values(l, m, lam, th))
                got = np.asarray(Y[(l, m)])
                max_err = max(max_err, float(np.max(np.abs(ref - got))))
        assert max_err < 1e-14

    def test_eval_sum_matches_manual_sum(self):
        # The accumulating reconstruction equals an explicit harmonic sum.
        lmax = 8
        lam = jnp.asarray(np.linspace(-np.pi, np.pi, 30))
        th = jnp.asarray(np.linspace(0.05, np.pi - 0.05, 30))
        rng = np.random.default_rng(0)
        coeff_map = {}
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                coeff_map[(l, m)] = float(rng.standard_normal())
        got = np.asarray(_sph_harmonic_eval_sum(coeff_map, lmax, lam, th))
        want = np.zeros_like(np.asarray(lam))
        for (l, m), a in coeff_map.items():
            want = want + a * np.asarray(_real_ylm_values(l, m, lam, th))
        assert float(np.max(np.abs(got - want))) < 1e-13

    def test_eval_sum_respects_sparse_map(self):
        # Only the (l, m) present in the map contribute.
        lmax = 6
        lam = jnp.asarray(np.linspace(-np.pi, np.pi, 20))
        th = jnp.asarray(np.linspace(0.1, np.pi - 0.1, 20))
        got = np.asarray(_sph_harmonic_eval_sum({(3, -2): 2.0}, lmax, lam, th))
        want = 2.0 * np.asarray(_real_ylm_values(3, -2, lam, th))
        assert float(np.max(np.abs(got - want))) < 1e-13


class TestNearZeroShortCircuit:
    def test_self_difference_is_exact_zero(self):
        f = Spherefun.from_function(
            lambda lam, th: jnp.cos(jnp.sin(lam) * jnp.sin(th)) + 0.3)
        d = f - f
        assert _snorm(d) == 0.0

    def test_genuine_difference_not_zeroed(self):
        # A real O(0.1) difference must survive the short-circuit.
        f = Spherefun.from_function(
            lambda lam, th: jnp.cos(jnp.sin(lam) * jnp.sin(th)))
        g = Spherefun.from_function(
            lambda lam, th: jnp.cos(jnp.sin(lam) * jnp.sin(th))
            + 0.5 * jnp.cos(th))
        d = g - f
        assert _snorm(d) > 0.1

    def test_small_real_difference_not_zeroed(self):
        # A genuine 1e-6-relative perturbation (well above the 1e-9 relative
        # zero threshold) must NOT be snapped to zero, and must reconstruct
        # with the right magnitude rather than chasing the operands' noise.
        f = Spherefun.from_function(
            lambda lam, th: jnp.cos(jnp.sin(lam) * jnp.sin(th)))
        g = Spherefun.from_function(
            lambda lam, th: jnp.cos(jnp.sin(lam) * jnp.sin(th))
            + 1e-6 * jnp.cos(th))
        d = g - f
        assert 1e-8 < _snorm(d) < 1e-4

    def test_product_of_orthogonal_basis_is_nonzero(self):
        # Sanity: a genuinely non-zero binary result is not short-circuited.
        f = Spherefun.from_function(lambda lam, th: jnp.cos(lam) * jnp.sin(th))
        assert _snorm(f * f) > 0.1
