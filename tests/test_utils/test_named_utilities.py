"""Core-suite coverage for the Fable 5 named-utilities sweep.

These exercise utils/cfpade.py, utils/polyops.py,
utils/fasttransforms.py, and utils/trigutils.py in the core test job
(the MATLAB-port tree runs in a separate CI job that does not count
toward the coverage gate).  The assertions mirror the corresponding
MATLAB-port tests at the same tolerances.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.utils.cfpade import _from_coeffs_any
from chebfunjax.utils.fasttransforms import dct, dlt, dst, idct, idlt, idst
from chebfunjax.utils.quadrature import trigpts

XS = jnp.asarray(np.linspace(-0.95, 0.95, 50))


class TestCf:
    def test_polynomial_cf(self):
        f = cj.chebfun(jnp.exp)
        _, _, _, lam = cj.cf(f, 2)
        assert abs(lam - 0.045017) < 1e-4

    def test_rational_cf(self):
        f = cj.chebfun(jnp.cos)
        p, q, r, _ = cj.cf(f, 1, 1)
        assert abs(float(p(jnp.asarray(0.3))) - 0.77015046914) < 1e-4
        # MATLAB pass(6): exp on [2, 6] with (5, 5)
        f6 = cj.chebfun(jnp.exp, domain=(2, 6))
        _, _, r6, _ = cj.cf(f6, 5, 5)
        xs6 = jnp.asarray(np.linspace(2, 6, 60))
        assert float(jnp.max(jnp.abs(f6(xs6) - r6(xs6)))) < 1e-6


class TestChebpade:
    def test_maehly(self):
        dom = (-1.0, 3.0)
        P = _from_coeffs_any(
            [0.5045, -1.3813, 2.1122, 0.0558, -0.6817], dom)
        Q = _from_coeffs_any(
            [1, 0.1155, -0.8573, -0.2679, 0.5246], dom)
        xs = jnp.asarray(np.linspace(-0.99, 2.99, 60))
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            R = cj.chebfun(lambda x: P(x) / Q(x), domain=dom)
        p, q, _ = cj.chebpade(R, 4, 4, "maehly")
        err = float(jnp.max(jnp.abs(p(xs) - P(xs)))) \
            + float(jnp.max(jnp.abs(q(xs) - Q(xs))))
        assert err < 1e-12

    def test_clenshaw_lord(self):
        f = cj.chebfun(jnp.exp)
        _, _, r = cj.chebpade(f, 2, 3)
        err = abs(float(np.real(np.asarray(
            r(jnp.asarray(0.5))))) - np.exp(0.5))
        assert err < 1e-5


class TestPolyops:
    def test_residue_roundtrip(self):
        f = cj.chebfun(lambda x: (x - 1.1) * (x ** 2 + 1) * (x - 10j))
        g = cj.chebfun(lambda x: x ** 5)
        r, p, k = cj.residue(g, f)
        pexact = np.array([10j, 1.1, -1j, 1j])
        pv = np.asarray(p)
        err = (np.linalg.norm(np.sort(pv.real) - np.sort(pexact.real))
               + np.linalg.norm(np.sort(pv.imag)
                                - np.sort(pexact.imag)))
        assert err < 1e-12
        B, A = cj.residue(np.array([1.0, 1.0]),
                          np.array([1.0, -1.0]), None)
        assert np.max(np.abs(np.asarray(B(XS))
                             - 2 * np.asarray(XS))) < 1e-13
        assert np.max(np.abs(np.asarray(A(XS))
                             - (np.asarray(XS) ** 2 - 1))) < 1e-13

    def test_fred_volt(self):
        def K(u, v):
            return jnp.exp(-((u - v) ** 2))

        f = cj.chebfun(jnp.sin)
        F = cj.fred(K, f)
        assert abs(float(F(jnp.asarray(0.5)))
                   - 0.293968048825243) < 1e-13
        V = cj.volt(K, f)
        assert abs(float(V(jnp.asarray(0.5)))
                   - (-0.013808570536509)) < 1e-13

    def test_poly(self):
        f = cj.chebfun(lambda x: 2 * x ** 2 - 3 * x + 1)
        c = np.asarray(cj.poly(f), dtype=float)
        np.testing.assert_allclose(c, [2.0, -3.0, 1.0], atol=1e-12)


class TestFasttransforms:
    def test_dct_dst_roundtrips(self):
        r = np.random.default_rng(0).standard_normal(17)
        for k in (1, 2, 3, 4):
            np.testing.assert_allclose(idct(dct(r, k), k), r,
                                       atol=1e-12)
            np.testing.assert_allclose(idst(dst(r, k), k), r,
                                       atol=1e-12)

    def test_dlt_roundtrip(self):
        r = np.random.default_rng(2).standard_normal(21)
        np.testing.assert_allclose(idlt(dlt(r)), r, atol=1e-12)


class TestTrigBary:
    def test_trig_and_nonequispaced(self):
        rng = np.random.default_rng(3453)
        xr = 2 * rng.random(400) - 1

        def p1(x):
            return (np.cos(4 * np.pi * x) - 2 * np.sin(3 * np.pi * x)
                    + 3 * np.sin(2 * np.pi * x)
                    - 2 * np.cos(np.pi * x) + 1)

        xk = np.asarray(trigpts(10, (-1, 1))[0])
        y = cj.trigBary(xr, p1(xk), xk, (-1, 1))
        assert np.max(np.abs(y - p1(xr))) < 1e-12

        # non-equispaced (first-kind Chebyshev) nodes, even count
        xk2 = -np.cos((2 * np.arange(1, 9) - 1) * np.pi / 16)

        def p2(x):
            return (2 * np.sin(3 * np.pi * x)
                    + 3 * np.sin(2 * np.pi * x)
                    - 2 * np.cos(np.pi * x) + 1)

        y2 = cj.trigBary(xr, p2(xk2), xk2, (-1, 1))
        assert np.max(np.abs(y2 - p2(xr))) < 1e-12


class TestSmallUtilities:
    def test_isSubset_and_nufft2(self):
        assert cj.isSubset([0, 2], [0, 2], 1e-12)
        assert not cj.isSubset([0, 2], [0, 1], 1e-12)
        rng = np.random.default_rng(0)
        M, N, m, n = 6, 7, 8, 5
        C = rng.random((m, n)) + 1j * rng.random((m, n))
        x = rng.random((M, N))
        y = rng.random((M, N))
        F = np.asarray(cj.nufft2(C, x, y))
        j, k = 2, 3
        direct = (np.exp(-2j * np.pi * y[j, k] * np.arange(m)) @ C
                  @ np.exp(-2j * np.pi * x[j, k] * np.arange(n)))
        assert abs(F[j, k] - direct) < 1e-12

    def test_hermpoly_lagpoly(self):
        xs = np.linspace(-1, 1, 40)
        h3 = cj.hermpoly(3)
        assert np.max(np.abs(np.asarray(h3(jnp.asarray(xs)))
                             - (8 * xs ** 3 - 12 * xs))) < 1e-12
        x01 = np.linspace(0, 1, 40)
        l2 = cj.lagpoly(2)
        assert np.max(np.abs(np.asarray(l2(jnp.asarray(x01)))
                             - (x01 ** 2 - 4 * x01 + 2) / 2)) < 1e-12

    def test_wronskian_and_prod(self):
        c = cj.chebfun(jnp.cos)
        s = cj.chebfun(jnp.sin)
        w = cj.wronskian(c, s)
        assert abs(float(w(jnp.asarray(0.3))) - 1.0) < 1e-11
        f = cj.chebfun(lambda x: x + 2)
        assert abs(float(f.prod()) - 27 * np.exp(-2)) < 1e-13


class TestTrigpade:
    def test_sin_and_rational(self):
        f = cj.chebfun(lambda t: jnp.sin(np.pi * t), trig=True)
        p, q, r = cj.trigpade(f, 1, 0)[:3]
        tt = jnp.asarray(np.linspace(-0.95, 0.95, 40))
        assert float(jnp.max(jnp.abs(f(tt) - r(tt)))) < 1e-12
        g = cj.chebfun(
            lambda t: 1.0 / (1.0 - 0.5 * jnp.cos(np.pi * t)),
            trig=True)
        _, _, r2 = cj.trigpade(g, 2, 2)[:3]
        assert float(jnp.max(jnp.abs(g(tt) - r2(tt)))) < 1e-12
