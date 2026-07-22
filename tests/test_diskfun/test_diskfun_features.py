"""Core (non-golden) tests for Diskfun feature methods added in the Fable 5
diskfun/diskfunv residue closure.

Covers: iszero, length, sample, minandmax2est, coeffs2diskfun (inverse of
coeffs2), partition/combine, vertcat, power (diskfun exponent), the
hyperbolic composition operators, and the definite/line integral method.
These mirror the MATLAB-port assertions but are kept independent of the
golden-reference .mat fixtures so they run in the fast (no-MATLAB) suite.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.diskfun.diskfun import Diskfun
from chebfunjax.diskfun.diskfunv import Diskfunv
from chebfunjax.domain import Domain

_EPS = float(jnp.finfo(jnp.float64).eps)
_TOL = 1000 * _EPS


def _df(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Diskfun.from_function(fn)


def _x(t, r):
    return r * jnp.cos(t)


def _y(t, r):
    return r * jnp.sin(t)


class TestIszero:
    def test_zero(self):
        assert _df(lambda t, r: 0.0 * r).iszero()

    def test_nonzero(self):
        assert not _df(lambda t, r: jnp.cos(_x(t, r))).iszero()

    def test_empty(self):
        assert Diskfun.empty().iszero()


class TestLength:
    def test_rank_le_min_slice_lengths(self):
        f = _df(lambda t, r: jnp.tanh(20 * _x(t, r)) * jnp.cos(50 * _x(t, r) * _y(t, r) + 1))
        m, n = f.length()
        assert f.rank <= min(m, n)

    def test_empty_length(self):
        assert Diskfun.empty().length() == (0, 0)


class TestSample:
    def test_shape_default(self):
        f = _df(lambda t, r: jnp.sin(jnp.pi * _x(t, r) * _y(t, r)))
        m, n = f.length()
        assert np.asarray(f.sample()).shape == (n, m)

    def test_values_equal_feval(self):
        f = _df(lambda t, r: jnp.sin(jnp.pi * _x(t, r) * _y(t, r)))
        m, ncols = 30, 21
        theta = -np.pi + 2.0 * np.pi * np.arange(m) / m
        r = np.cos(np.pi * (ncols - 1 - np.arange(ncols)) / (2 * ncols - 2))
        tt, rr = np.meshgrid(theta, r)
        ref = np.asarray(f(jnp.asarray(tt), jnp.asarray(rr)))
        assert np.linalg.norm(ref - np.asarray(f.sample(m, ncols)), np.inf) < 100 * _EPS

    def test_constant_all_ones(self):
        f = _df(lambda t, r: 1.0 + 0.0 * r)
        assert np.linalg.norm(np.asarray(f.sample(64, 64)) - 1.0, np.inf) < 100 * _EPS

    def test_bad_inputs(self):
        f = _df(lambda t, r: 1.0 + 0.0 * r)
        with pytest.raises(ValueError):
            f.sample(0, 10)
        with pytest.raises(ValueError):
            f.sample(10, 0)


class TestMinandmax2est:
    def test_x_subset(self):
        f = _df(lambda t, r: _x(t, r))
        mM = np.asarray(f.minandmax2est())
        assert mM.shape == (2,)
        assert mM[0] >= -1.0 - _TOL and mM[1] <= 1.0 + _TOL
        assert mM[0] < -0.9 and mM[1] > 0.9

    def test_diskfunv_range(self):
        F = Diskfunv.from_functions(lambda t, r: _x(t, r), lambda t, r: _y(t, r))
        rng = np.asarray(F.minandmax2est())
        assert rng.shape == (4,)
        assert rng[0] >= -1.0 - _TOL and rng[1] <= 1.0 + _TOL
        assert rng[2] >= -1.0 - _TOL and rng[3] <= 1.0 + _TOL


class TestCoeffs2diskfun:
    def test_inverse_of_coeffs2(self):
        f = _df(lambda t, r: r**3 * jnp.cos(3 * t) + r**2 * jnp.sin(t) ** 2)
        g = Diskfun.coeffs2diskfun(f.coeffs2())
        assert float((f - g).norm()) < _TOL

    def test_explicit_matrix(self):
        c = 1j / 2.0 * np.array([[0, 0, 0], [1, 0, -1]], dtype=np.complex128)
        f = _df(lambda t, r: r * jnp.sin(t))
        assert float((f - Diskfun.coeffs2diskfun(c)).norm()) < _TOL

    def test_zero_scalar(self):
        assert Diskfun.coeffs2diskfun(0).iszero()


class TestPartitionCombine:
    def _fe(self, t, r):
        return jnp.sin(jnp.pi * _x(t, r) * _y(t, r))

    def _fo(self, t, r):
        return jnp.sin(jnp.pi * _x(t, r))

    def test_partition_mixed(self):
        f = _df(lambda t, r: self._fe(t, r) + self._fo(t, r))
        feven, fodd = f.partition()
        assert float((_df(self._fe) - feven).norm()) < _TOL
        assert float((_df(self._fo) - fodd).norm()) < _TOL

    def test_combine_roundtrip(self):
        f = _df(lambda t, r: self._fe(t, r) + self._fo(t, r))
        g = Diskfun.combine(_df(self._fe), _df(self._fo))
        assert float((f - g).norm()) < _TOL

    def test_combine_parity_error(self):
        f = _df(lambda t, r: self._fe(t, r) + self._fo(t, r))
        with pytest.raises(ValueError, match="parity"):
            Diskfun.combine(f, f)


class TestVertcat:
    def test_two_components(self):
        f = _df(lambda t, r: jnp.cos(_x(t, r)))
        g = _df(lambda t, r: jnp.sin(_y(t, r)))
        F = Diskfun.vertcat(f, g)
        assert isinstance(F, Diskfunv)
        assert float((F.components[0] - f).norm()) < _TOL
        assert float((F.components[1] - g).norm()) < _TOL

    def test_three_errors(self):
        f = _df(lambda t, r: jnp.cos(_x(t, r)))
        with pytest.raises((ValueError, TypeError)):
            Diskfun.vertcat(f, f, f)


class TestPowerAndComposition:
    def test_diskfun_power_diskfun(self):
        def _c(t, r):
            return jnp.cos(_x(t, r) * _y(t, r))

        f = _df(_c)
        g = _df(lambda t, r: _c(t, r) ** _c(t, r))
        assert float(((f**f) - g).norm()) < _TOL

    def test_cosh_sinh(self):
        f = _df(lambda t, r: jnp.cos(_x(t, r) * _y(t, r)))
        gc = _df(lambda t, r: jnp.cosh(jnp.cos(_x(t, r) * _y(t, r))))
        gs = _df(lambda t, r: jnp.sinh(jnp.cos(_x(t, r) * _y(t, r))))
        assert float((f.cosh() - gc).norm()) < _TOL
        assert float((f.sinh() - gs).norm()) < _TOL

    def test_compose_chebfun(self):
        f = _df(lambda t, r: _x(t, r) + _y(t, r))
        g = Chebfun.from_function(lambda t: t**2, domain=Domain((-2.0, 2.0)))
        h = f.compose(g)
        h_true = _df(lambda t, r: (_x(t, r) + _y(t, r)) ** 2)
        assert float((h - h_true).norm()) < _TOL


class TestIntegral:
    def test_double_integral_matches_sum(self):
        f = _df(lambda t, r: jnp.exp(-(_x(t, r) ** 2) - _y(t, r) ** 2))
        assert abs(float(f.integral()) - float(f.sum())) < _TOL

    def test_empty(self):
        assert float(Diskfun.empty().integral()) == 0.0

    def test_unitcircle_radial(self):
        # Radial g = exp(-2 r^2): boundary integral = exp(-2) * 2*pi.
        g = _df(lambda t, r: jnp.exp(-2 * (_x(t, r) ** 2 + _y(t, r) ** 2)))
        ref = np.exp(-2.0) * 2.0 * np.pi
        assert abs(float(g.integral("unitcircle")) - ref) < _TOL

    def test_bad_string(self):
        f = _df(lambda t, r: 1.0 + 0.0 * r)
        with pytest.raises(ValueError):
            f.integral("nonsense")
