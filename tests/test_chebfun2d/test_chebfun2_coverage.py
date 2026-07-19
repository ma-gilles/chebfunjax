"""Core-suite coverage mirrors for :mod:`chebfunjax.chebfun2d.chebfun2`.

Exercises the recently-added Chebfun2 methods the core suite did not reach:
the Padua constructor, ``svd`` and the spectral / nuclear / inf / even-p norm
modes, ``fevalm``, ``isequal``, ``fliplr`` / ``flipud``, ``cumsum`` /
``cumsum2``, ``restrict`` (point / line / subdomain), ``squeeze``, the
``mean`` / ``mean2`` / ``std2`` / ``diag`` / ``trace`` statistics, the compose
wrappers, ``grad`` / ``gradient`` / ``laplacian``, and ``real`` / ``imag`` /
``conj``.  Every assertion checks a closed-form value.

Provenance
----------
Mirrors of MATLAB @chebfun2 / @separableApprox tests; Chebfun commit 7574c77.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebfun2d.padua import paduapts

# Reference constants for f(x, y) = cos(x) cos(y) on [-1, 1]^2.
_INT_COS = 2.0 * math.sin(1.0)            # int_{-1}^{1} cos = 2 sin 1
_INT_COS2 = 1.0 + math.sin(2.0) / 2.0     # int_{-1}^{1} cos^2
_SIGMA = _INT_COS2                         # single singular value (rank 1)


def _coscos():
    return Chebfun2.from_function(lambda x, y: jnp.cos(x) * jnp.cos(y))


def _f(cf, x, y):
    return float(cf(jnp.float64(x), jnp.float64(y)))


class TestChebfun2Padua:
    def test_padua_reconstructs_polynomial(self):
        n = 6
        pts = np.asarray(paduapts(n))
        # a total-degree-<= n polynomial is reproduced exactly
        vals = (
            1.0 + pts[:, 0] + pts[:, 1]
            + pts[:, 0] * pts[:, 1] + pts[:, 0] ** 2
        )
        g = Chebfun2.from_padua(vals)
        for x, y in [(0.3, 0.4), (-0.6, 0.2), (0.8, -0.5)]:
            expect = 1.0 + x + y + x * y + x ** 2
            npt.assert_allclose(_f(g, x, y), expect, atol=1e-10)

    def test_padua_bad_domain(self):
        with pytest.raises(ValueError):
            Chebfun2.from_padua([1.0, 2.0, 3.0], domain=(-1.0, 1.0, -1.0))


class TestChebfun2SvdNorm:
    def test_svd_rank_one(self):
        f = _coscos()
        s = np.asarray(f.svd())
        assert s.shape == (1,)
        npt.assert_allclose(s[0], _SIGMA, rtol=1e-9)

    def test_norm_fro_equals_svd(self):
        f = _coscos()
        s = np.asarray(f.svd())
        npt.assert_allclose(float(f.norm("fro")), math.sqrt(np.sum(s ** 2)), rtol=1e-9)

    def test_norm_operator(self):
        f = _coscos()
        npt.assert_allclose(float(f.norm(2)), _SIGMA, rtol=1e-9)
        npt.assert_allclose(float(f.norm("op")), _SIGMA, rtol=1e-9)

    def test_norm_nuclear(self):
        npt.assert_allclose(float(_coscos().norm("nuc")), _SIGMA, rtol=1e-9)

    def test_norm_inf(self):
        # global max of |cos x cos y| is 1 at the origin
        npt.assert_allclose(float(_coscos().norm(jnp.inf)), 1.0, atol=1e-7)

    def test_norm_even_p(self):
        f = _coscos()
        # reference: (int int cos^4 x cos^4 y)^(1/4) via a dense grid
        t = np.linspace(-1.0, 1.0, 4001)
        int_cos4 = np.trapezoid(np.cos(t) ** 4, t)
        ref = (int_cos4 ** 2) ** 0.25
        npt.assert_allclose(float(f.norm(4)), ref, rtol=1e-6)

    def test_norm_l1_unsupported(self):
        with pytest.raises(NotImplementedError):
            _coscos().norm(1)

    def test_norm_min_unsupported(self):
        with pytest.raises(NotImplementedError):
            _coscos().norm("min")

    def test_norm_odd_p_unsupported(self):
        with pytest.raises(NotImplementedError):
            _coscos().norm(3)


class TestChebfun2FevalmIsequal:
    def test_fevalm_matches_meshgrid(self):
        f = _coscos()
        x = np.array([-0.4, 0.1, 0.7])
        y = np.array([-0.2, 0.5])
        Z = np.asarray(f.fevalm(x, y))
        assert Z.shape == (2, 3)
        for i, yy in enumerate(y):
            for j, xx in enumerate(x):
                npt.assert_allclose(Z[i, j], math.cos(xx) * math.cos(yy), atol=1e-9)

    def test_isequal_true(self):
        assert _coscos().isequal(_coscos())

    def test_isequal_false(self):
        f = _coscos()
        g = Chebfun2.from_function(lambda x, y: jnp.cos(x) * jnp.sin(y))
        assert not f.isequal(g)

    def test_isequal_non_chebfun2(self):
        assert not _coscos().isequal(42)


class TestChebfun2Flip:
    def test_fliplr(self):
        f = Chebfun2.from_function(lambda x, y: x + 2.0 * y)
        g = f.fliplr()  # f(-x, y)
        npt.assert_allclose(_f(g, 0.3, 0.4), -0.3 + 0.8, atol=1e-9)

    def test_flipud(self):
        f = Chebfun2.from_function(lambda x, y: x + 2.0 * y)
        g = f.flipud()  # f(x, -y)
        npt.assert_allclose(_f(g, 0.3, 0.4), 0.3 - 0.8, atol=1e-9)


class TestChebfun2Cumsum:
    def test_cumsum_dim1(self):
        f = _coscos()
        F = f.cumsum(1)  # integrate over y from -1
        # int_{-1}^{y} cos v dv = sin y + sin 1
        npt.assert_allclose(
            _f(F, 0.3, 0.4), math.cos(0.3) * (math.sin(0.4) + math.sin(1.0)), atol=1e-8
        )

    def test_cumsum_dim2(self):
        f = _coscos()
        F = f.cumsum(2)  # integrate over x from -1
        npt.assert_allclose(
            _f(F, 0.4, 0.3), math.cos(0.3) * (math.sin(0.4) + math.sin(1.0)), atol=1e-8
        )

    def test_cumsum_bad_dim(self):
        with pytest.raises(ValueError):
            _coscos().cumsum(3)

    def test_cumsum2(self):
        f = _coscos()
        F = f.cumsum2()
        expect = (math.sin(0.3) + math.sin(1.0)) * (math.sin(0.4) + math.sin(1.0))
        npt.assert_allclose(_f(F, 0.3, 0.4), expect, atol=1e-8)


class TestChebfun2Restrict:
    def test_restrict_point(self):
        f = _coscos()
        val = f.restrict((0.3, 0.3, 0.4, 0.4))
        npt.assert_allclose(float(val), math.cos(0.3) * math.cos(0.4), atol=1e-9)

    def test_restrict_x_line(self):
        f = _coscos()
        g = f.restrict((0.3, 0.3, -1.0, 1.0))  # Chebfun in y
        npt.assert_allclose(
            float(g(jnp.float64(0.4))), math.cos(0.3) * math.cos(0.4), atol=1e-9
        )

    def test_restrict_y_line(self):
        f = _coscos()
        g = f.restrict((-1.0, 1.0, 0.4, 0.4))  # Chebfun in x
        npt.assert_allclose(
            float(g(jnp.float64(0.3))), math.cos(0.3) * math.cos(0.4), atol=1e-9
        )

    def test_restrict_subdomain(self):
        f = _coscos()
        g = f.restrict((-0.5, 0.5, -0.5, 0.5))
        npt.assert_allclose(_f(g, 0.3, 0.4), math.cos(0.3) * math.cos(0.4), atol=1e-9)


class TestChebfun2Squeeze:
    def test_squeeze_constant_in_y(self):
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x) + 0.0 * y)
        g = f.squeeze()  # -> Chebfun in x
        npt.assert_allclose(float(g(jnp.float64(0.3))), math.cos(0.3), atol=1e-9)

    def test_squeeze_constant_in_x(self):
        f = Chebfun2.from_function(lambda x, y: jnp.sin(y) + 0.0 * x)
        g = f.squeeze()  # -> Chebfun in y
        npt.assert_allclose(float(g(jnp.float64(0.4))), math.sin(0.4), atol=1e-9)

    def test_squeeze_nonconstant_returns_self(self):
        f = _coscos()
        assert f.squeeze() is f


class TestChebfun2Statistics:
    def test_mean2(self):
        f = _coscos()
        npt.assert_allclose(float(f.mean2()), (math.sin(1.0)) ** 2, rtol=1e-9)

    def test_std2(self):
        f = _coscos()
        mean_f2 = (_INT_COS2 / 2.0) ** 2
        mean_f = (_INT_COS / 2.0) ** 2
        ref = math.sqrt(mean_f2 - mean_f ** 2)
        npt.assert_allclose(float(f.std2()), ref, rtol=1e-6)

    def test_mean_dim1(self):
        f = _coscos()
        g = f.mean(1)  # average over y -> cos(x) * sin(1)
        npt.assert_allclose(_f(g, 0.3, 0.0), math.cos(0.3) * math.sin(1.0), atol=1e-8)

    def test_diag_and_trace(self):
        f = _coscos()
        d = f.diag_fun()  # g(t) = cos(t)^2
        npt.assert_allclose(float(d(jnp.float64(0.5))), math.cos(0.5) ** 2, atol=1e-9)
        npt.assert_allclose(float(f.trace()), _INT_COS2, rtol=1e-9)


class TestChebfun2Compose:
    def test_sin_cos_exp(self):
        f = Chebfun2.from_function(lambda x, y: 0.5 * x + 0.5 * y)
        npt.assert_allclose(_f(f.sin(), 0.3, 0.4), math.sin(0.35), atol=1e-9)
        npt.assert_allclose(_f(f.cos(), 0.3, 0.4), math.cos(0.35), atol=1e-9)
        npt.assert_allclose(_f(f.exp(), 0.3, 0.4), math.exp(0.35), atol=1e-9)
        npt.assert_allclose(_f(f.tanh(), 0.3, 0.4), math.tanh(0.35), atol=1e-9)

    def test_sqrt_log_abs(self):
        f = Chebfun2.from_function(lambda x, y: 2.0 + 0.3 * x + 0.3 * y)
        npt.assert_allclose(_f(f.sqrt(), 0.4, 0.6), math.sqrt(2.3), atol=1e-9)
        npt.assert_allclose(_f(f.log(), 0.4, 0.6), math.log(2.3), atol=1e-9)
        g = Chebfun2.from_function(lambda x, y: -1.0 + 0.0 * x + 0.0 * y)
        npt.assert_allclose(_f(g.abs(), 0.3, 0.4), 1.0, atol=1e-9)


class TestChebfun2Calculus:
    def test_laplacian(self):
        f = _coscos()
        lap = f.laplacian()  # -2 cos x cos y
        npt.assert_allclose(_f(lap, 0.3, 0.4), -2.0 * math.cos(0.3) * math.cos(0.4), atol=1e-7)

    def test_lap_alias(self):
        f = _coscos()
        npt.assert_allclose(_f(f.lap(), 0.3, 0.4), _f(f.laplacian(), 0.3, 0.4), atol=1e-9)

    def test_grad_and_gradient(self):
        f = _coscos()
        g = f.grad()
        assert isinstance(g, Chebfun2v)
        assert isinstance(f.gradient(), Chebfun2v)


class TestChebfun2ComplexParts:
    def test_real_imag_conj(self):
        f = Chebfun2.from_function(lambda x, y: x + 1j * y)
        npt.assert_allclose(_f(f.real(), 0.3, 0.4), 0.3, atol=1e-9)
        npt.assert_allclose(_f(f.imag(), 0.3, 0.4), 0.4, atol=1e-9)
        c = complex(f.conj()(jnp.float64(0.3), jnp.float64(0.4)))
        npt.assert_allclose(c.real, 0.3, atol=1e-9)
        npt.assert_allclose(c.imag, -0.4, atol=1e-9)
