"""Core-suite coverage mirrors for :mod:`chebfunjax.spherefun.spherefunv`.

Exercises the empty object, the vector algebra (cross, dot, norm, times,
power, scalar multiply), and the single-level surface differential operators
(curl, divergence/div, vorticity/vort, normal, tangent) on cheap fields.
Nested compositions (e.g. div(curl(F))) are deliberately avoided — they blow
up XLA CPU compilation.

Provenance
----------
Mirrors of MATLAB @spherefunv tests; Chebfun commit 7574c77.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy.testing as npt
import pytest

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

PT = (jnp.float64(0.3), jnp.float64(0.7))


def _const(v):
    return Spherefun.from_function(lambda lam, th, _v=v: jnp.full_like(lam, _v))


def _ex():
    return Spherefunv(_const(1.0), _const(0.0), _const(0.0))


def _ey():
    return Spherefunv(_const(0.0), _const(1.0), _const(0.0))


class TestSpherefunvEmpty:
    def test_empty_flag(self):
        e = Spherefunv.empty()
        assert e.isempty()

    def test_nonempty_flag(self):
        assert not _ex().isempty()

    def test_empty_dot(self):
        assert _ex().dot(Spherefunv.empty()).isempty()

    def test_empty_cross(self):
        assert Spherefunv.empty().cross(_ex()).isempty()

    def test_empty_curl(self):
        assert Spherefunv.empty().curl().isempty()

    def test_empty_divergence(self):
        assert Spherefunv.empty().divergence().isempty()

    def test_empty_normal(self):
        assert Spherefunv.empty().normal().isempty()

    def test_empty_tangent_returns_self(self):
        e = Spherefunv.empty()
        assert e.tangent() is e

    def test_empty_vorticity(self):
        assert Spherefunv.empty().vorticity().isempty()


class TestSpherefunvConstruction:
    def test_bad_component_count_raises(self):
        with pytest.raises(ValueError):
            Spherefunv(_const(1.0))
        with pytest.raises(ValueError):
            Spherefunv(_const(1.0), _const(1.0), _const(1.0), _const(1.0))

    def test_repr(self):
        r = repr(_ex())
        assert "Spherefunv with 3 components" in r
        assert "[0]" in r


class TestSpherefunvAlgebra:
    def test_cross_of_basis(self):
        c = _ex().cross(_ey())  # e_x x e_y = e_z
        vals = [float(comp(*PT)) for comp in c.components]
        npt.assert_allclose(vals, [0.0, 0.0, 1.0], atol=1e-9)

    def test_cross_requires_three(self):
        two = Spherefunv(_const(1.0), _const(0.0))
        with pytest.raises(ValueError):
            two.cross(two)

    def test_dot_orthogonal_and_self(self):
        npt.assert_allclose(float(_ex().dot(_ey())(*PT)), 0.0, atol=1e-9)
        npt.assert_allclose(float(_ex().dot(_ex())(*PT)), 1.0, atol=1e-9)

    def test_norm_three_component(self):
        f = Spherefunv(_const(1.0), _const(0.0), _const(0.0))
        npt.assert_allclose(float(f.norm()(*PT)), 1.0, atol=1e-9)

    def test_norm_two_component(self):
        f = Spherefunv(_const(3.0), _const(4.0))
        npt.assert_allclose(float(f.norm()(*PT)), 5.0, atol=1e-9)

    def test_times_vector(self):
        p = _ex().times(_ex())  # componentwise square
        vals = [float(c(*PT)) for c in p.components]
        npt.assert_allclose(vals, [1.0, 0.0, 0.0], atol=1e-9)

    def test_times_scalar(self):
        p = _ex().times(2.0)
        npt.assert_allclose(float(p.components[0](*PT)), 2.0, atol=1e-9)

    def test_power(self):
        p = Spherefunv(_const(2.0), _const(3.0), _const(0.0)).power(2)
        vals = [float(c(*PT)) for c in p.components]
        npt.assert_allclose(vals, [4.0, 9.0, 0.0], atol=1e-9)

    def test_scalar_mul_both_sides(self):
        left = (2.0 * _ex()).components[0]
        right = (_ex() * 2.0).components[0]
        npt.assert_allclose(float(left(*PT)), 2.0, atol=1e-9)
        npt.assert_allclose(float(right(*PT)), 2.0, atol=1e-9)


class TestSpherefunvDifferential:
    def test_divergence_of_unit_normal(self):
        # surface divergence of the outward unit normal = 2 on the unit sphere
        d = Spherefunv.unormal().divergence()
        for lam, th in [(0.3, 0.7), (1.1, 1.9), (-0.5, 2.2)]:
            npt.assert_allclose(
                float(d(jnp.float64(lam), jnp.float64(th))), 2.0, atol=1e-8
            )

    def test_div_alias_matches_divergence(self):
        N = Spherefunv.unormal()
        npt.assert_allclose(float(N.div()(*PT)), float(N.divergence()(*PT)), atol=1e-10)

    def test_curl_of_unit_normal_is_zero(self):
        c = Spherefunv.unormal().curl()
        for comp in c.components:
            npt.assert_allclose(float(comp(*PT)), 0.0, atol=1e-8)

    def test_vorticity_of_unit_normal_is_zero(self):
        npt.assert_allclose(float(Spherefunv.unormal().vorticity()(*PT)), 0.0, atol=1e-8)

    def test_vort_alias_matches_vorticity(self):
        N = Spherefunv.unormal()
        npt.assert_allclose(float(N.vort()(*PT)), float(N.vorticity()(*PT)), atol=1e-10)

    def test_normal_of_unit_normal_is_itself(self):
        # projecting the normal field onto the normal direction is the identity
        N = Spherefunv.unormal()
        proj = N.normal()
        for orig, pr in zip(N.components, proj.components):
            npt.assert_allclose(float(pr(*PT)), float(orig(*PT)), atol=1e-8)

    def test_tangent_of_unit_normal_is_zero(self):
        # the normal field has no tangential part
        t = Spherefunv.unormal().tangent()
        for comp in t.components:
            npt.assert_allclose(float(comp(*PT)), 0.0, atol=1e-8)

    def test_require3_error(self):
        two = Spherefunv(_const(1.0), _const(0.0))
        with pytest.raises(ValueError):
            two.divergence()
