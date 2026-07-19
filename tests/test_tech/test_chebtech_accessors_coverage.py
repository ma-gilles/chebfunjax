"""Core-suite coverage mirrors for the Chebtech accessor helpers.

Exercises ``alias`` / ``angles`` / ``sample`` / ``trigcoeffs`` / ``values`` on
both :class:`Chebtech1` and :class:`Chebtech2`, plus the ``turbo`` contour-
integral constructor, all with closed-form assertions.

Provenance
----------
Mirrors of MATLAB @chebtech(1,2) accessor tests; Chebfun commit 7574c77.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.tech.chebtech import (
    Chebtech1,
    Chebtech2,
    chebpts,
)


@pytest.mark.parametrize("cls, kind", [(Chebtech2, 2), (Chebtech1, 1)])
class TestChebtechAccessors:
    def test_sample_default_matches_values(self, cls, kind):
        f = cls.from_function(jnp.cos)
        vals, pts = f.sample()
        # sampling on the native grid reproduces coeffs2vals
        npt.assert_allclose(np.asarray(vals), np.asarray(f.values), rtol=1e-12)
        # and the points are the native grid
        npt.assert_allclose(np.asarray(pts), np.asarray(chebpts(len(f), kind=kind)), atol=1e-12)

    def test_sample_upsample(self, cls, kind):
        f = cls.from_function(jnp.cos)
        n = len(f) + 7
        vals, pts = f.sample(n)
        npt.assert_allclose(np.asarray(vals), np.cos(np.asarray(pts)), atol=1e-11)

    def test_sample_downsample_polynomial(self, cls, kind):
        # low-degree polynomial: aliasing to any n >= degree+1 is exact
        f = cls.from_function(lambda x: 1.0 + 2.0 * x + 3.0 * x ** 2)
        vals, pts = f.sample(4)
        expect = 1.0 + 2.0 * np.asarray(pts) + 3.0 * np.asarray(pts) ** 2
        npt.assert_allclose(np.asarray(vals), expect, atol=1e-11)

    def test_alias_pad(self, cls, kind):
        f = cls.from_function(jnp.cos)
        c = f.coeffs
        padded = cls.alias(c, c.shape[0] + 3)
        assert padded.shape[0] == c.shape[0] + 3
        npt.assert_allclose(np.asarray(padded[: c.shape[0]]), np.asarray(c), atol=1e-14)
        npt.assert_allclose(np.asarray(padded[c.shape[0] :]), 0.0, atol=1e-14)

    def test_alias_to_one(self, cls, kind):
        f = cls.from_function(jnp.cos)
        aliased = cls.alias(f.coeffs, 1)
        assert aliased.shape[0] == 1

    def test_angles_matches_acos_grid(self, cls, kind):
        n = 9
        ang = np.asarray(cls.angles(n))
        expect = np.arccos(np.asarray(chebpts(n, kind=kind)))
        npt.assert_allclose(np.sort(ang), np.sort(expect), atol=1e-12)

    def test_angles_zero(self, cls, kind):
        assert cls.angles(0).shape[0] == 0

    def test_trigcoeffs_constant(self, cls, kind):
        f = cls.from_function(lambda x: jnp.ones_like(x))
        c = np.asarray(f.trigcoeffs(5))
        mid = c.shape[0] // 2
        npt.assert_allclose(abs(c[mid]), 1.0, atol=1e-9)
        rest = np.delete(np.abs(c), mid)
        npt.assert_allclose(rest, 0.0, atol=1e-9)

    def test_trigcoeffs_even_length(self, cls, kind):
        f = cls.from_function(lambda x: jnp.ones_like(x))
        c = np.asarray(f.trigcoeffs(4))
        assert c.shape[0] == 4

    def test_values_property(self, cls, kind):
        f = cls.from_function(jnp.sin)
        pts = chebpts(len(f), kind=kind)
        npt.assert_allclose(np.asarray(f.values), np.sin(np.asarray(pts)), atol=1e-11)


class TestChebtech2StaticTransforms:
    def test_vals2coeffs_roundtrip(self):
        pts = chebpts(12, kind=2)
        vals = jnp.sin(pts)
        c = Chebtech2.vals2coeffs(vals)
        back = Chebtech2.coeffs2vals(c)
        npt.assert_allclose(np.asarray(back), np.asarray(vals), atol=1e-12)

    def test_alias_static_downsample_exact_for_low_degree(self):
        # coeffs of 3 x^2 - 1 aliased down should still evaluate the polynomial
        f = Chebtech2.from_function(lambda x: 3.0 * x ** 2 - 1.0)
        aliased = Chebtech2.alias(f.coeffs, 3)
        vals = Chebtech2.coeffs2vals(aliased)
        pts = chebpts(3, kind=2)
        npt.assert_allclose(np.asarray(vals), 3.0 * np.asarray(pts) ** 2 - 1.0, atol=1e-11)


class TestChebtechTurbo:
    def test_turbo_real(self):
        f = Chebtech2.from_function(jnp.cos, turbo=True)
        for x in (-0.7, -0.1, 0.3, 0.8):
            npt.assert_allclose(float(f(jnp.float64(x))), np.cos(x), atol=1e-11)

    def test_turbo_pure_imaginary(self):
        f = Chebtech2.from_function(lambda x: 1j * jnp.exp(x), turbo=True)
        for x in (-0.5, 0.2, 0.6):
            val = complex(f(jnp.float64(x)))
            npt.assert_allclose(val.imag, np.exp(x), atol=1e-10)
            npt.assert_allclose(val.real, 0.0, atol=1e-10)

    def test_turbo_complex(self):
        f = Chebtech2.from_function(lambda x: jnp.exp(x) + 1j * jnp.sin(x), turbo=True)
        for x in (-0.4, 0.1, 0.7):
            val = complex(f(jnp.float64(x)))
            npt.assert_allclose(val.real, np.exp(x), atol=1e-10)
            npt.assert_allclose(val.imag, np.sin(x), atol=1e-10)


@pytest.mark.parametrize("cls, kind", [(Chebtech2, 2), (Chebtech1, 1)])
class TestChebtechBary:
    def test_bary_reproduces_polynomial(self, cls, kind):
        # A degree-5 polynomial sampled on 8 points is reproduced exactly.
        p = lambda x: 1.0 + x - 2.0 * x**3 + 0.5 * x**5  # noqa: E731
        gvals = p(chebpts(8, kind=kind))
        x = jnp.linspace(-1.0, 1.0, 31, dtype=jnp.float64)
        npt.assert_allclose(np.asarray(cls.bary(x, gvals)), np.asarray(p(x)),
                            atol=5e-15)

    def test_bary_at_nodes(self, cls, kind):
        gvals = jnp.sin(3.0 * chebpts(12, kind=kind))
        out = cls.bary(chebpts(12, kind=kind), gvals)
        npt.assert_allclose(np.asarray(out), np.asarray(gvals), atol=1e-14)

    def test_barywts_matches_generic(self, cls, kind):
        from chebfunjax.utils.interpolation import bary_weights

        n = 9
        v = np.asarray(cls.barywts(n), dtype=np.float64)
        w = np.asarray(bary_weights(chebpts(n, kind=kind)), dtype=np.float64)
        # Barycentric weights are scale-invariant (incl. overall sign);
        # normalize both by their first entry before comparing.
        npt.assert_allclose(v / v[0], w / w[0], atol=1e-12)
