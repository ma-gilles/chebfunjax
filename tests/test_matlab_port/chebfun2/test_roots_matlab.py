"""Port of MATLAB Chebfun tests/chebfun2/test_roots.m (Fable 5).

FIXED (Fable 5): Chebfun2.roots now returns complex-valued Chebfun zero
curves (marching squares + complex-Newton polish; rank-1 lines exact).

The MATLAB assertions fall into two groups:
  * rank-1 line curves (pass 1-5): compared against the exact line
    parametrization at tol = 1e-12;
  * higher-rank curves (pass 6-25): compared through the
    parametrization-invariant arc length ``sum(abs(diff(c)))`` and enclosed
    area ``sum(real(c).*diff(imag(c)))`` at MATLAB's tolerances.

``roots`` returns a Python list of curves (chebfunjax's stand-in for a
MATLAB quasimatrix), so quasimatrix reductions are computed by iterating.

Provenance
----------
MATLAB source : tests/chebfun2/test_roots.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d import chebfun2

TOL = 1e-12    # rank-1 curves
TOL2 = 1e-8    # rank >1 curves, disjoint
TOL3 = 1e-4    # rank >1 curves, noncircular
TOL4 = 1e-2    # intersecting curves, not properly disentangled

_T = jnp.asarray(np.linspace(-1.0, 1.0, 200))


def _arclengths(curves):
    """Per-curve arc length sum(abs(diff(c)))."""
    return sorted(float(abs(c.diff()).sum()) for c in curves)


def _total_arclength(curves):
    return sum(float(abs(c.diff()).sum()) for c in curves)


def _area(curves):
    """abs(sum(real(c).*diff(imag(c)))) summed over curves."""
    return abs(sum(float((c.real() * c.imag().diff()).sum())
                   for c in curves))


def _matches(curves, exact_fns, tol):
    """Every exact curve is matched (in either orientation) by some
    returned curve to within ``tol`` in the sup norm over t."""
    ok = True
    for e in exact_fns:
        ev = np.asarray(e(_T))
        best = np.inf
        for c in curves:
            cv = np.asarray(c(_T))
            d1 = float(np.max(np.abs(cv - ev)))
            d2 = float(np.max(np.abs(cv[::-1] - ev)))
            best = min(best, d1, d2)
        ok = ok and (best < tol)
    return ok and (len(curves) == len(exact_fns))


class TestChebfun2Roots:
    def test_rank1_lines(self):
        # pass(1): 1/2 - y  ->  horizontal line  x + 1i/2
        f = chebfun2(lambda x, y: 0.5 - y)
        assert _matches(f.roots(), [lambda t: t + 0.5j], TOL)

        # pass(2): 1/2 - x  ->  vertical line  1i*x + 1/2
        f = chebfun2(lambda x, y: 0.5 - x)
        assert _matches(f.roots(), [lambda t: 1j * t + 0.5], TOL)

        # pass(3): x*y  ->  [x, 1i*x]
        f = chebfun2(lambda x, y: x * y)
        assert _matches(f.roots(),
                        [lambda t: t + 0j, lambda t: 1j * t], TOL)

        # pass(4): cos(5*pi*x)  ->  vertical lines at each x-root.
        f = chebfun2(lambda x, y: jnp.cos(5 * np.pi * x))
        xroots = (2 * np.arange(-5, 5, 1) + 1) / 10.0  # +-0.1,...,+-0.9
        exact = [(lambda t, s=s: s + 1j * t) for s in xroots]
        assert _matches(f.roots(), exact, TOL)

        # pass(5): x*(y-1/2) on [-2,2,-3,5].
        f = chebfun2(lambda x, y: x * (y - 0.5), domain=(-2, 2, -3, 5))
        exact = [lambda t: 2 * t + 0.5j,
                 lambda t: 1j * (4 * (t + 1) - 3)]
        assert _matches(f.roots(), exact, TOL)

    def test_circle_arclength_and_area(self):
        # pass(6,7): unit-quarter circle radius 1/2.
        f = chebfun2(lambda x, y: x ** 2 + y ** 2 - 0.25)
        c = f.roots()
        assert abs(_total_arclength(c) - np.pi) < TOL2
        assert abs(_area(c) - np.pi / 4) < TOL2

    def test_ellipse(self):
        # pass(8,9): eccentric ellipse.
        f = chebfun2(lambda x, y: x ** 2 + (10 * y) ** 2 - 0.25)
        c = f.roots()
        assert abs(_total_arclength(c) - 2.031987090050447) < TOL3
        assert abs(_area(c) - np.pi / 40) < TOL3

    def test_shifted_circles(self):
        # pass(10): (x-1)^2 + y^2 - 1/4  (half circle, clipped by boundary).
        f = chebfun2(lambda x, y: (x - 1) ** 2 + y ** 2 - 0.25)
        assert abs(_total_arclength(f.roots()) - np.pi / 2) < TOL2

        # pass(11): (x-1)^2 + (y+1)^2 - 1/4  (quarter circle).
        f = chebfun2(lambda x, y: (x - 1) ** 2 + (y + 1) ** 2 - 0.25)
        assert abs(_total_arclength(f.roots()) - np.pi / 4) < TOL2

    def test_unit_circumference(self):
        # pass(12): radius 1/pi circle has circumference 2.
        r = 1 / np.pi
        f = chebfun2(lambda x, y: x ** 2 + y ** 2 - r ** 2)
        assert abs(_total_arclength(f.roots()) - 2.0) < TOL2

    def test_two_disjoint_circles(self):
        # pass(13): two well-separated circles, each circumference 2.
        r = 1 / np.pi
        f = chebfun2(lambda x, y: (x ** 2 + y ** 2 - r ** 2)
                     * ((x - 0.6) ** 2 + (y - 0.5) ** 2 - r ** 2))
        al = _arclengths(f.roots())
        assert np.max(np.abs(np.array(al) - 2.0)) < TOL2

    def test_close_circles_total(self):
        # pass(14): closer circles; total circumference ~4 (tol4).
        r = 1 / np.pi
        f = chebfun2(lambda x, y: (x ** 2 + y ** 2 - r ** 2)
                     * ((x - 0.2) ** 2 + (y - 0.5) ** 2 - r ** 2))
        assert abs(_total_arclength(f.roots()) - 4.0) < TOL4

    def test_circle_times_exp(self):
        # pass(15): circle * (exp(x)-1.2); total arclength ~4 (tol4).
        r = 1 / np.pi
        f = chebfun2(lambda x, y: (x ** 2 + y ** 2 - r ** 2)
                     * (jnp.exp(x) - 1.2))
        assert abs(_total_arclength(f.roots()) - 4.0) < TOL4

    def test_concentric_circles(self):
        # pass(16): radii 1/2 and sqrt(0.3).
        f = chebfun2(lambda x, y: (x ** 2 + y ** 2 - 0.25)
                     * (x ** 2 + y ** 2 - 0.300))
        al = _arclengths(f.roots())
        want = sorted([np.pi, np.pi * np.sqrt(0.300 / 0.250)])
        assert np.max(np.abs(np.array(al) - np.array(want))) < TOL2

        # pass(17): radii 1/2 and sqrt(0.26).
        f = chebfun2(lambda x, y: (x ** 2 + y ** 2 - 0.25)
                     * (x ** 2 + y ** 2 - 0.260))
        al = _arclengths(f.roots())
        want = sorted([np.pi, np.pi * np.sqrt(0.260 / 0.250)])
        assert np.max(np.abs(np.array(al) - np.array(want))) < TOL2

    def test_two_circles_custom_domain(self):
        # pass(18): two unit-quarter circles on a wide domain.
        f = chebfun2(lambda x, y: (x ** 2 + y ** 2 - 0.25)
                     * ((x - 1.1) ** 2 + (y - 3) ** 2 - 0.25),
                     domain=(-1.3, 1.1, -0.9, 3))
        assert abs(_total_arclength(f.roots()) - 1.25 * np.pi) < TOL2

    def test_scaled_function(self):
        # pass(19,20): huge scaling does not change the zero set.
        f = chebfun2(lambda x, y: 1e100 * (x ** 2 + y ** 2 - 0.25))
        c = f.roots()
        assert abs(_total_arclength(c) - np.pi) < TOL2
        assert abs(_area(c) - np.pi / 4) < TOL2

    def test_sine_curve(self):
        # pass(21): y - sin(x) on [-pi, pi] x [-1.1, 1.3].
        f = chebfun2(lambda x, y: y - jnp.sin(x),
                     domain=(-np.pi, np.pi, -1.1, 1.3))
        assert abs(_total_arclength(f.roots())
                   - 7.640395578055424035) < TOL2

    def test_tanh_curve(self):
        # pass(22): x - tanh(y).
        f = chebfun2(lambda x, y: x - jnp.tanh(y))
        assert abs(_total_arclength(f.roots()) - 2.53182746833076) < TOL2

    def test_parabola(self):
        # pass(23): y - x^2.
        f = chebfun2(lambda x, y: y - x ** 2)
        assert abs(_total_arclength(f.roots())
                   - 2.957885715089195) < TOL2

    def test_hyperbola(self):
        # pass(24): y^2 - x^2 - 0.1 (two branches).
        f = chebfun2(lambda x, y: y ** 2 - x ** 2 - 0.1,
                     domain=(-1, 1, -1.5, 1.5))
        al = _arclengths(f.roots())
        assert np.max(np.abs(np.array(al) - 2.51829399795912)) < TOL2

    def test_cross(self):
        # pass(25): y^2 - x^2 (two lines through the origin), tol4.
        f = chebfun2(lambda x, y: y ** 2 - x ** 2)
        al = _arclengths(f.roots())
        assert np.max(np.abs(np.array(al) - np.sqrt(8))) < TOL4
