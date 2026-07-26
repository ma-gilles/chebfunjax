"""Core unit tests for the chebtech surface added in the xfail-flip pass.

Mirrors the MATLAB-port coverage of ``sign``, ``poly``,
``extractBoundaryRoots``, ``chebTcoeffs2chebUcoeffs``, ``roots(qz=True)`` and
the Chebtech1 ``compose``/``restrict``/``min``/``max``/``minandmax`` methods,
plus the Chebtech1 complex-scalar addition dtype promotion.  These live
outside ``tests/test_matlab_port`` so the new source lines are covered by the
core suite as well.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestNewChebtechSurface:
    def test_sign_real(self, Tech):
        f = Tech.from_function(lambda x: jnp.sin(x) + 2.0)
        assert float((f.sign() - 1.0).norm(jnp.inf)) < 10 * EPS
        g = Tech.from_function(lambda x: -(jnp.sin(x) + 2.0))
        assert float((g.sign() + 1.0).norm(jnp.inf)) < 10 * EPS

    def test_sign_complex(self, Tech):
        f = Tech.from_function(lambda x: jnp.exp(1j * jnp.pi * x))
        assert float((f.sign() - f).norm(jnp.inf)) < 1e2 * EPS

    def test_poly_quadratic(self, Tech):
        # 4 x^2 - 2 x + 3.7 -> [4, -2, 3.7] (highest degree first).
        f = Tech.from_function(lambda x: 4 * x ** 2 - 2 * x + 3.7)
        assert _ninf(f.poly() - jnp.array([4.0, -2.0, 3.7])) < 10 * f.vscale * EPS

    def test_extract_boundary_roots(self, Tech):
        # (1+x)^2 (1-x) exp(x): l == 2, r == 1.
        f = Tech.from_function(
            lambda x: jnp.exp(x) * ((1 + x) ** 2) * (1 - x)
        )
        g, l, r = f.extractBoundaryRoots()
        gex = Tech.from_function(lambda x: jnp.exp(x))
        xs = jnp.asarray(np.linspace(-0.9, 0.9, 50))
        assert l == 2 and r == 1
        assert _ninf(g(xs) - gex(xs)) < (1e2 ** 3) * EPS

    def test_chebT2U_roundtrip(self, Tech):
        cT = jnp.array([1.875, 1.75, 1.0, 0.25, 0.125])
        cU = Tech.chebTcoeffs2chebUcoeffs(cT)
        assert _ninf(cU - jnp.array([1.375, 0.75, 0.4375, 0.125, 0.0625])) < 10 * EPS

    def test_roots_qz(self, Tech):
        f = Tech.from_function(lambda x: (x - 0.5) * (x + 0.25))
        r = np.sort(np.asarray(f.roots(qz=True)))
        assert r.size == 2
        assert _ninf(f(jnp.asarray(r))) < 1e2 * EPS

    def test_roots_complex_flag(self, Tech):
        # Core mirror of tests/test_matlab_port/chebtech/test_roots.py
        # pass(n, 6): roots(1 + 25 x^2, 'complex', 1) -> +/- i/5.
        f = Tech.from_function(lambda x: 1 + 25 * x**2)
        r = np.asarray(f.roots(complex_roots=True))
        r = r[np.argsort(np.imag(r))[::-1]]
        assert r.size == 2
        assert _ninf(r - np.array([1j, -1j]) / 5) < 10 * EPS
        # Plain roots() finds no real root in [-1, 1] for this function.
        assert np.asarray(f.roots()).size == 0

    def test_roots_recurse_toggle(self, Tech):
        # recurse=False solves one colleague problem (no subdivision); the
        # real roots of sin(100 pi x) in [-1, 1] are k/100, k=-100..100.
        f = Tech.from_function(lambda x: jnp.sin(100 * jnp.pi * x))
        r = np.sort(np.asarray(f.roots(recurse=False)))
        assert r.size == 201
        assert _ninf(r - np.arange(-100, 101) / 100) < 1e3 * EPS

    def test_min_max(self, Tech):
        f = Tech.from_function(lambda x: jnp.sin(10 * x))
        ymin, _ = f.min()
        ymax, _ = f.max()
        assert abs(float(ymax) - 1.0) < 1e2 * EPS
        assert abs(float(ymin) + 1.0) < 1e2 * EPS

    def test_complex_scalar_add_keeps_imag(self, Tech):
        # Chebtech1.__add__ dtype promotion (kept for Chebtech2 too).
        f = Tech.from_function(lambda x: jnp.sin(x))
        alpha = -0.19 + 0.075j
        g = f + alpha
        xs = jnp.asarray(np.linspace(-1, 1, 40))
        assert _ninf(g(xs) - (jnp.sin(xs) + alpha)) < 10 * g.vscale * EPS


def test_chebtech1_compose_restrict():
    # Chebtech1-specific method mirrors of the Chebtech2 methods.
    f = Chebtech1.from_function(lambda x: x)
    g = f.compose(jnp.sin)
    h = Chebtech1.from_function(jnp.sin)
    n = max(g.n, h.n)
    gc = np.zeros(n)
    gc[: g.n] = np.asarray(g.coeffs)
    hc = np.zeros(n)
    hc[: h.n] = np.asarray(h.coeffs)
    assert _ninf(gc - hc) < 10 * h.vscale * EPS

    r = Chebtech1.from_function(lambda x: jnp.exp(x)).restrict(-0.2, 0.4)
    xs = jnp.asarray(np.linspace(-0.2, 0.4, 50))
    mapx = (2.0 / 0.6) * (xs + 0.2) - 1.0
    assert _ninf(jnp.exp(xs) - r(mapx)) < 1e3 * r.vscale * EPS


def test_unhappy_propagation():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = Chebtech2.from_function(lambda x: jnp.sqrt(x + 1))
    f = Chebtech2.from_function(lambda x: jnp.cos(x + 1))
    assert not g.ishappy
    assert not (f + g).ishappy
    assert not (f * g).ishappy
