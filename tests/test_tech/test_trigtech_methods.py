"""Core-mirror coverage for trigtech methods added in the xfail-flip wave
(Opus 4.8[1m]): sign, poly, iszero, isnan/isinf/isfinite, real/imag, qr,
mldivide/mrdivide, circconv, horzcat, minandmax, roots('complex').

These are fast smoke checks (smooth functions only) that exercise the new
source lines directly; the faithful MATLAB parity assertions live in
tests/test_matlab_port/trigtech/.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)


def _tt(f):
    return Trigtech.from_function(f)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestTrigtechMethods:
    def test_sign_real_and_complex(self):
        f = _tt(lambda x: 2 + jnp.sin(jnp.pi * x))
        xx = jnp.linspace(-0.9, 0.9, 20)
        assert _ninf(f.sign()(xx) - 1) < 1e-13
        g = _tt(lambda x: jnp.exp(1j * jnp.pi * x))
        assert _ninf(g.sign()(xx) - g(xx)) < 1e-12

    def test_poly_is_coeffs_transpose(self):
        f = _tt(lambda x: 1 + jnp.cos(jnp.pi * x))
        assert _ninf(f.poly() - jnp.array([0.5, 1.0, 0.5])) < 1e-13
        assert Trigtech.empty().poly().size == 0

    def test_iszero_and_predicates(self):
        z = Trigtech.from_values(jnp.zeros(3))
        assert bool(z.iszero())
        f = _tt(lambda x: jnp.cos(jnp.pi * x))
        assert not f.isnan() and not f.isinf() and f.isfinite()
        assert f.isreal()
        y = jnp.ones(11).at[3].set(jnp.inf)
        fi = Trigtech.from_values(y)
        assert fi.isinf() and not fi.isfinite()
        fn = Trigtech.from_values(jnp.array([jnp.nan]))
        assert fn.isnan()

    def test_real_imag_exact(self):
        f = _tt(lambda x: jnp.cos(jnp.pi * x) + 1j * jnp.sin(jnp.pi * x))
        r = f.real()
        i = f.imag()
        xx = jnp.linspace(-0.9, 0.9, 20)
        assert _ninf(r(xx) - jnp.cos(jnp.pi * xx)) < 1e-12
        assert _ninf(i(xx) - jnp.sin(jnp.pi * xx)) < 1e-12

    def test_size_and_vscale_columns(self):
        f = _tt(lambda x: jnp.stack(
            [jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x)], axis=-1))
        assert f.size(2) == 2 and f.num_columns == 2
        assert f.vscale_columns().shape == (2,)

    def test_qr_orthonormal(self):
        f = _tt(lambda x: jnp.stack(
            [jnp.exp(jnp.sin(jnp.pi * x)), 3.0 / (4 - jnp.cos(jnp.pi * x))],
            axis=-1))
        Q, R, E = f.qr(want_e=True)
        assert _ninf(jnp.asarray(Q.innerProduct(Q)) - jnp.eye(2)) < 1e-13
        xx = jnp.linspace(-1, 1, 30, endpoint=False)
        assert _ninf(((Q @ R) - (f @ E))(xx)) < 1e-12

    def test_qr_single_column(self):
        f = _tt(lambda x: jnp.exp(jnp.sin(jnp.pi * x)))
        Q, R = f.qr()
        assert abs(float(jnp.asarray(Q.innerProduct(Q))) - 1.0) < 1e-13

    def test_mldivide(self):
        f = _tt(lambda x: jnp.stack(
            [jnp.ones_like(x), jnp.cos(jnp.pi * x), jnp.sin(jnp.pi * x)],
            axis=-1))
        g = _tt(lambda x: jnp.cos(jnp.pi * x))
        X = Trigtech.mldivide(f, g)
        assert _ninf(jnp.asarray(X).ravel() - jnp.array([0.0, 1.0, 0.0])) < 1e-12
        with pytest.raises(ValueError, match="mldivide"):
            Trigtech.mldivide(f, 2)

    def test_mrdivide(self):
        f = _tt(lambda x: jnp.stack(
            [jnp.exp(jnp.sin(jnp.pi * x)), 3.0 / (4 - jnp.cos(jnp.pi * x))],
            axis=-1))
        assert Trigtech.mrdivide(f, 0).isnan()
        g = Trigtech.mrdivide(f, 2.0)
        xx = jnp.linspace(-1, 1, 30, endpoint=False)
        assert _ninf(jnp.asarray(g(xx)) - jnp.asarray(f(xx)) / 2) < 1e-12
        with pytest.raises(ValueError, match="trigtechDivTrigtech"):
            Trigtech.mrdivide(f, f)
        with pytest.raises(ValueError, match="badArg"):
            Trigtech.mrdivide(f, True)
        with pytest.raises(ValueError, match="size"):
            Trigtech.mrdivide(f, jnp.array([[1.0, 2.0, 3.0]]))

    def test_circconv(self):
        f = _tt(lambda x: jnp.tanh(jnp.cos(jnp.pi * x)))
        g = f.circconv(f)
        approx = jnp.asarray(g(jnp.array([0.0]))).ravel()[0]
        exact = jnp.asarray((f * f).sum()).ravel()[0]
        assert abs(complex(approx) - complex(exact)) < 1e-12
        assert f.circconv(Trigtech.empty()).isempty()

    def test_horzcat(self):
        a = _tt(lambda x: jnp.sin(jnp.pi * x))
        b = _tt(lambda x: jnp.cos(jnp.pi * x))
        c = Trigtech.horzcat(a, Trigtech.empty(), b)
        assert c.num_columns == 2
        xx = jnp.linspace(-0.9, 0.9, 15)
        assert _ninf(jnp.asarray(c(xx))[:, 0] - jnp.sin(jnp.pi * xx)) < 1e-12

    def test_minandmax_smooth(self):
        f = _tt(lambda x: jnp.exp(-jnp.cos(2 * jnp.pi * x)))
        (mn, _), (mx, _) = f.minandmax()
        assert abs(float(mn) - np.exp(-1)) < 1e-12
        assert abs(float(mx) - np.exp(1)) < 1e-12

    def test_roots_complex_flag(self):
        f = _tt(lambda x: 2 + jnp.cos(jnp.pi * x))
        r = f.roots(complex=True)
        assert r.size >= 1
        # No real roots (2 + cos never vanishes on the real line).
        assert f.roots().size == 0

    def test_extract_column(self):
        f = _tt(lambda x: jnp.stack(
            [jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x)], axis=-1))
        c0 = f.extract_column(0)
        xx = jnp.linspace(-0.9, 0.9, 15)
        assert _ninf(c0(xx) - jnp.sin(jnp.pi * xx)) < 1e-12
