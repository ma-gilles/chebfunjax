"""Core-suite tests for the spherical-harmonic-basis surface gradient (Fable 5).

These back the harmonic-basis gradient that replaced the value-space
tangential-derivative route inside :meth:`Spherefun.gradient` /
:meth:`Spherefun.grad`.  The analytic Cartesian surface-gradient recurrence is
applied to every ``Y_l^m``, so the three components are exactly tangential
(``x*fx + y*fy + z*fz == 0`` to machine precision) with no ``1/sin(theta)``
amplification.  The ``f +/- 0`` structural short-circuit is what lets
``tangent(grad f) == grad f`` hold at the ``1e2*eps`` tolerance.

Provenance
----------
Mirrors of MATLAB @spherefun/gradient.m behaviour; Chebfun commit 7574c77.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import (
    Spherefun,
    _complex_to_real_sph,
    _real_to_complex_sph,
    _sph_grad_op_minus,
    _sph_grad_op_plus,
    _sph_grad_op_z,
    _spherefun_grad_harmonic,
)

LAM = jnp.asarray(np.linspace(-3.0, 3.0, 11))
TH = jnp.asarray(np.linspace(0.2, 2.9, 11))
LL, TT = jnp.meshgrid(LAM, TH, indexing="ij")


def _gv(sf):
    return np.asarray(sf(LL, TT))


class TestCoeffTransform:
    def test_real_complex_roundtrip_is_identity(self):
        rng = np.random.default_rng(0)
        lmax = 6
        a = {}
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                a[(l, m)] = float(rng.standard_normal())
        c = _real_to_complex_sph(a, lmax)
        back = _complex_to_real_sph(c, lmax)
        err = max(abs(a[k] - back[k]) for k in a)
        assert err < 1e-14

    def test_complex_coeffs_satisfy_reality(self):
        # A real field's complex coeffs obey c_{l,-m} = (-1)^m conj(c_{l,m}).
        rng = np.random.default_rng(1)
        lmax = 5
        a = {(l, m): float(rng.standard_normal())
             for l in range(lmax + 1) for m in range(-l, l + 1)}
        c = _real_to_complex_sph(a, lmax)
        for l in range(lmax + 1):
            for mu in range(1, l + 1):
                lhs = c[(l, -mu)]
                rhs = ((-1) ** mu) * np.conj(c[(l, mu)])
                assert abs(lhs - rhs) < 1e-14


class TestGradOperators:
    def test_operator_sqrt_args_nonnegative(self):
        # Every admissible (l, m) yields real coefficients (no nan/inf).
        for l in range(0, 12):
            for m in range(-l, l + 1):
                for op in (_sph_grad_op_z, _sph_grad_op_plus,
                           _sph_grad_op_minus):
                    for co in op(l, m).values():
                        assert np.isfinite(co)

    def test_z_operator_matches_known_recurrence(self):
        # d/dz Y_l^m -> -l A_l^m Y_{l+1}^m + (l+1) A_{l-1}^m Y_{l-1}^m.
        l, m = 3, 1
        o = _sph_grad_op_z(l, m)
        a_l = np.sqrt(((l + 1) ** 2 - m * m) / ((2 * l + 1) * (2 * l + 3)))
        a_lm1 = np.sqrt((l * l - m * m) / ((2 * l - 1) * (2 * l + 1)))
        assert abs(o[(l + 1, m)] - (-l * a_l)) < 1e-14
        assert abs(o[(l - 1, m)] - ((l + 1) * a_lm1)) < 1e-14


class TestHarmonicGradient:
    def test_gradient_of_harmonic_is_tangential(self):
        # Each Y_l^m has an exactly tangential surface gradient.
        for l, m in [(1, 0), (2, 1), (3, -2), (4, 3)]:
            f = Spherefun.sphharm(l, m)
            fx, fy, fz = _spherefun_grad_harmonic(f)
            x = np.cos(np.asarray(LL)) * np.sin(np.asarray(TT))
            y = np.sin(np.asarray(LL)) * np.sin(np.asarray(TT))
            z = np.cos(np.asarray(TT))
            normal = np.max(np.abs(x * _gv(fx) + y * _gv(fy) + z * _gv(fz)))
            assert normal < 1e-12

    def test_gradient_of_harmonic_matches_dirichlet_energy(self):
        # int |grad Y_l^m|^2 = l(l+1) (the surface Dirichlet energy).
        l, m = 3, 2
        fx, fy, fz = Spherefun.sphharm(l, m).grad()
        e = fx * fx + fy * fy + fz * fz
        assert abs(float(e.sum()) - l * (l + 1)) < 1e-8

    def test_general_gradient_is_tangential(self):
        f = Spherefun.from_function(
            lambda lam, th: jnp.cos(jnp.cos(lam) * jnp.sin(th))
            * jnp.sin(2 * jnp.cos(th)))
        fx, fy, fz = _spherefun_grad_harmonic(f)
        x = np.cos(np.asarray(LL)) * np.sin(np.asarray(TT))
        y = np.sin(np.asarray(LL)) * np.sin(np.asarray(TT))
        z = np.cos(np.asarray(TT))
        normal = np.max(np.abs(x * _gv(fx) + y * _gv(fy) + z * _gv(fz)))
        assert normal < 1e-12

    def test_gradient_of_constant_is_zero(self):
        f = Spherefun.from_function(lambda lam, th: jnp.ones_like(lam * th))
        for comp in _spherefun_grad_harmonic(f):
            assert float(jnp.max(jnp.abs(_gv(comp)))) < 1e-12


class TestExactZeroShortCircuit:
    def test_subtract_exact_zero_returns_self(self):
        f = Spherefun.from_function(
            lambda lam, th: jnp.cos(lam) * jnp.sin(th))
        z = Spherefun.from_function(
            lambda lam, th: jnp.zeros_like(lam * th))
        assert z._is_exact_zero()
        # f - 0 must be f exactly (no re-approximation drift).
        diff = np.max(np.abs(_gv(f - z) - _gv(f)))
        assert diff == 0.0
        # 0 + f and f + 0 too.
        assert np.max(np.abs(_gv(f + z) - _gv(f))) == 0.0
        assert np.max(np.abs(_gv(z + f) - _gv(f))) == 0.0

    def test_nonzero_field_not_flagged_zero(self):
        f = Spherefun.from_function(
            lambda lam, th: 1e-10 * jnp.cos(lam) * jnp.sin(th))
        assert not f._is_exact_zero()
