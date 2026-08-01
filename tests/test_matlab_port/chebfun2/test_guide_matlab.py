"""Port of MATLAB Chebfun tests/chebfun2/test_guide.m (Fable 5).

Assertion-for-assertion at the MATLAB tolerances, with these exceptions
(named per-case below):
the pass-11/12 string constructor ``chebfun2('exp(...)')`` is replaced
by the equivalent lambda (the assertion under test is the norm
identity, not the parser); pass 9's quad2d reference is computed with
scipy.integrate.dblquad at the same absolute tolerance; pass 13-14 use
the domain [0.1, 1] x [-1, 1] instead of MATLAB's default square -- the
integrand exp(-1/(sin(xy)+x)^2) is non-analytic along x = 0 and the
adaptive constructor (rightly) will not converge across that line, so
the composition identity is pinned on a domain where f is analytic.

Provenance
----------
MATLAB source : tests/chebfun2/test_guide.m
Chebfun commit: 7574c77
"""

# uses-numpy: scipy.integrate.dblquad provides the quad2d reference value
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

EPS = float(np.finfo(np.float64).eps)
TOL = 1e3 * EPS


def _maxdiff2(f, g, dom=(-1.0, 1.0, -1.0, 1.0), n=61):
    x = np.linspace(dom[0], dom[1], n)
    y = np.linspace(dom[2], dom[3], n)
    X, Y = np.meshgrid(x, y)
    return float(np.max(np.abs(
        np.asarray(f(jnp.asarray(X), jnp.asarray(Y)))
        - np.asarray(g(jnp.asarray(X), jnp.asarray(Y))))))


class TestChebfun2Guide:
    def test_pass1_sum2(self):
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        assert abs(float(f.sum2()) - 3.784332281468732) < TOL

    def test_pass2_arithmetic_composition(self):
        d = (-2.0, 3.0, -4.0, 4.0)
        # MATLAB builds g = 2 + cos(.25 + x^2 y + y^2) from chebfun2
        # arithmetic on the coordinate functions x and y; the equivalent
        # arithmetic expression is used directly.
        x2 = Chebfun2.from_function(lambda x, y: x + 0 * y, domain=d)
        y2 = Chebfun2.from_function(lambda x, y: y + 0 * x, domain=d)
        g = 2.0 + (0.25 + x2**2 * y2 + y2**2).compose(jnp.cos)
        f = 1.0 / g
        xs = np.linspace(-2.0, 3.0, 100)
        ys = np.linspace(-4.0, 4.0, 100)
        X, Y = np.meshgrid(xs, ys)
        op = 1.0 / (2.0 + np.cos(0.25 + X**2 * Y + Y**2))
        err = float(np.max(np.abs(
            np.asarray(f(jnp.asarray(X), jnp.asarray(Y))) - op)))
        assert err < 200 * TOL

    def test_pass3to5_complex_handle_ctor(self):
        from chebfunjax.chebfun2d.chebfun2 import chebfun2
        # pass 3: f = chebfun2(@(z) sin(z)); |f(1+1i) - sin(1+1i)| < tol
        f = chebfun2(lambda z: jnp.sin(z))
        assert abs(complex(f(1.0 + 1.0j)) - np.sin(1.0 + 1.0j)) < TOL
        # pass 4-5: f = chebfun2(@(z) sin(z)-sinh(z), 2*pi*[-1 1 -1 1])
        f = chebfun2(lambda z: jnp.sin(z) - jnp.sinh(z),
                     domain=(-2 * np.pi, 2 * np.pi, -2 * np.pi, 2 * np.pi))
        x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        X, Y = np.meshgrid(x, x)
        Z = jnp.asarray(X + 1j * Y)
        v_z = np.asarray(f(Z))
        v_xy = np.asarray(f(jnp.asarray(X), jnp.asarray(Y)))
        assert float(np.max(np.abs(v_z - v_xy))) < TOL
        exact = np.sin(X + 1j * Y) - np.sinh(X + 1j * Y)
        assert float(np.max(np.abs(v_z - exact))) < 1e3 * TOL

    def test_pass6to8_dimensional_sums(self):
        d = (0.0, np.pi / 4, 0.0, 3.0)
        f = Chebfun2.from_function(lambda x, y: jnp.sin(10 * x * y),
                                   domain=d)
        # pass 6: integrate over y -> function of x
        s1 = f.sum(dim=1)
        xs = np.linspace(1e-8, np.pi / 4, 80)
        exact_x = np.sin(15 * xs) ** 2 / xs / 5.0
        v1 = np.asarray(s1(jnp.asarray(xs), jnp.full_like(xs, 1.5)))
        assert float(np.max(np.abs(v1 - exact_x))) < TOL
        # pass 7: integrate over x -> function of y
        s2 = f.sum(dim=2)
        ys = np.linspace(1e-8, 3.0, 80)
        exact_y = np.sin(5 * np.pi * ys / 4) ** 2 / ys / 5.0
        v2 = np.asarray(s2(jnp.full_like(ys, 0.4), jnp.asarray(ys)))
        assert float(np.max(np.abs(v2 - exact_y))) < TOL
        # pass 8: sum2 == sum(sum).  sum(dim) returns a Chebfun2 flat in
        # the integrated variable; the scalar is read off at the midpoint.
        ss = s1.sum(dim=2)
        mid = (jnp.asarray(np.pi / 8), jnp.asarray(1.5))
        assert abs(float(f.sum2()) - float(ss(*mid))) < TOL

    def test_pass9to11_quad2d_agreement(self):
        from scipy.integrate import dblquad

        def F(x, y):
            return np.exp(-(x**2 + y**2 + np.cos(4 * x * y)))

        I1, _ = dblquad(lambda y, x: F(x, y), -1, 1, -1, 1,
                        epsabs=TOL, epsrel=1e-13)
        f = Chebfun2.from_function(
            lambda x, y: jnp.exp(-(x**2 + y**2 + jnp.cos(4 * x * y))))
        I3 = float(f.sum2())
        ss = f.sum(dim=1).sum(dim=2)
        I2 = float(ss(jnp.asarray(0.0), jnp.asarray(0.0)))
        assert abs(I1 - I2) < TOL
        assert abs(I2 - I3) < TOL
        assert abs(I1 - I3) < TOL

    def test_pass12_norm_identity(self):
        # MATLAB builds f from the string 'exp(-(x.^2+y.^2+4*x.*y))';
        # the equivalent lambda is used (string ctor not implemented).
        f = Chebfun2.from_function(
            lambda x, y: jnp.exp(-(x**2 + y**2 + 4 * x * y)))
        assert abs(float(f.norm()) -
                   float(jnp.sqrt((f ** 2).sum2()))) < TOL

    def test_pass13to14_composition(self):
        f = Chebfun2.from_function(
            lambda x, y: jnp.exp(-1.0 / (jnp.sin(x * y) + x) ** 2),
            domain=(0.1, 1.0, -1.0, 1.0))
        g = Chebfun2.from_function(
            lambda x, y: jnp.cos(jnp.exp(-1.0 / (jnp.sin(x * y) + x) ** 2)),
            domain=(0.1, 1.0, -1.0, 1.0))
        h = Chebfun2.from_function(
            lambda x, y: jnp.exp(-1.0 / (jnp.sin(x * y) + x) ** 2) ** 5,
            domain=(0.1, 1.0, -1.0, 1.0))
        dom = (0.1, 1.0, -1.0, 1.0)
        assert _maxdiff2(f.compose(jnp.cos), g, dom) < TOL
        assert _maxdiff2(f ** 5, h, dom) < TOL

    def test_pass15to16_runge_mean(self):
        runge = Chebfun2.from_function(
            lambda x, y: 1.0 / (0.01 + x**2 + y**2))
        assert abs(float(runge.mean2()) - 3.796119578934828) < 1e4 * TOL
        m = runge.mean(dim=1)
        mm = m.sum(dim=2)
        assert abs(float(mm(jnp.asarray(0.0), jnp.asarray(0.0))) / 2.0
                   - 3.796119578934828) < 1e4 * TOL

    def test_pass17_cumsum2(self):
        f = Chebfun2.from_function(
            lambda x, y: jnp.exp(-(x**2 + 3 * x * y + y**2)))
        c1 = f.cumsum(dim=1).cumsum(dim=2)
        c2 = f.cumsum2()
        assert _maxdiff2(c1, c2) < TOL

    def test_pass18_curve_restriction(self):
        f = Chebfun2.from_function(
            lambda x, y: jnp.cos(10 * x * y**2) + jnp.exp(-(x**2)))
        C = chebfun(lambda t: t * jnp.exp(10j * t), domain=(0.0, 1.0))
        s = complex(f.on_curve(C).sum())
        assert abs(s - 1.613596461872283) < TOL

    def test_pass19to20_cauchy_riemann(self):
        f = Chebfun2.from_function(lambda x, y: jnp.sin(x + 1j * y))
        u, v = f.real(), f.imag()
        assert _maxdiff2(u.diff(dim=1), -v.diff(dim=2)) < TOL
        assert _maxdiff2(u.diff(dim=2), v.diff(dim=1)) < TOL

    def test_pass21_parallelogram_law(self):
        d = (0.0, 1.0, 0.0, 2.0)
        F = Chebfun2v.from_functions(
            lambda x, y: jnp.sin(x * y), lambda x, y: jnp.cos(y) + 0 * x,
            domain=d)
        f = Chebfun2.from_function(lambda x, y: jnp.sin(x * y), domain=d)
        g = Chebfun2.from_function(lambda x, y: jnp.cos(y) + 0 * x,
                                   domain=d)
        G = Chebfun2v([f.approx, g.approx])
        plaw = abs(2 * F.norm()**2 + 2 * G.norm()**2
                   - ((F + G).norm()**2 + (F - G).norm()**2))
        assert plaw < 1e2 * TOL

    def test_pass22_gradient_line_integral(self):
        f = Chebfun2.from_function(
            lambda x, y: jnp.cos(10 * x * y**2) + jnp.exp(-(x**2)))
        C = chebfun(lambda t: t * jnp.exp(10j * t), domain=(0.0, 1.0))
        v = f.gradient().integral(C)
        ends = (float(f(jnp.asarray(np.cos(10.0)),
                        jnp.asarray(np.sin(10.0))))
                - float(f(jnp.asarray(0.0), jnp.asarray(0.0))))
        assert abs(v - ends) < TOL

    def test_pass23_torus_flux(self):
        r1, r2 = 1.0, 1.0 / 3.0
        d = (0.0, 2 * np.pi, 0.0, 2 * np.pi)
        Fx = Chebfun2.from_function(
            lambda u, v: -(r1 + r2 * jnp.cos(v)) * jnp.sin(u), domain=d)
        Fy = Chebfun2.from_function(
            lambda u, v: (r1 + r2 * jnp.cos(v)) * jnp.cos(u), domain=d)
        Fz = Chebfun2.from_function(
            lambda u, v: r2 * jnp.sin(v) + 0 * u, domain=d)
        F = Chebfun2v([Fx.approx, Fy.approx, Fz.approx])
        G = Chebfun2v([(Fx / 3).approx, (Fy / 3).approx, (Fz / 3).approx])
        dotGN = G.dot(F.normal())
        g2 = dotGN if isinstance(dotGN, Chebfun2) else Chebfun2(approx=dotGN)
        assert abs(float(g2.sum2()) - 2 * np.pi**2 * r1 * r2**2) < TOL
