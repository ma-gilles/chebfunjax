"""Port of MATLAB Chebfun tests/diskfun/test_integral.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_integral.m
Chebfun commit: 7574c77

Cartesian ``@(x,y)`` handles are written in polar coordinates
``(theta, r)`` with ``x = r cos(theta)``, ``y = r sin(theta)``.  The MATLAB
reference quantities ``sum(f(:,1))`` etc. (definite integrals of boundary /
radial slices) are reproduced with independent spectral / analytic values.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
from scipy.special import erf

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.diskfun.diskfun import Diskfun
from chebfunjax.domain import Domain

_EPS = float(jnp.finfo(jnp.float64).eps)
_TOL = 1000 * _EPS


def _df(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Diskfun.from_function(fn)


class TestDiskfunIntegral:
    def test_empty_double_integral(self):
        # pass(1): integral(empty) == 0
        assert float(Diskfun.empty().integral()) == 0.0

    def test_empty_line_integral(self):
        # pass(2): integral(empty, chebfun) == 0
        c = Chebfun.from_function(lambda x: x, domain=Domain((-1.0, 1.0)))
        assert float(Diskfun.empty().integral(c)) == 0.0

    def test_unitcircle_boundary(self):
        # pass(3): abs(sum(f(:,1)) - integral(f, 'unitcircle')) < tol
        f = _df(lambda t, r: jnp.sin((r * jnp.cos(t)) ** 2 + 3 * r * jnp.sin(t)))
        # Reference: integral over theta of the boundary trace f(theta, 1).
        boundary = Chebfun.from_function(
            lambda th: f(th, jnp.ones_like(th)), domain=Domain((-np.pi, np.pi))
        )
        ref = float(boundary.sum())
        got = float(f.integral("unitcircle"))
        assert abs(ref - got) < _TOL

    def test_unitcircle_neumann_harmonic(self):
        # pass(4): abs(integral(harmonic(3, 2, 'neumann'), 'unitcircle')) < tol
        h = Diskfun.harmonic(3, 2, "neumann")
        assert abs(float(h.integral("unitcircle"))) < _TOL

    def test_line_integral_circle(self):
        # pass(5): z = .5 exp(i pi x); abs(.5*sum(g(:,.5)) - integral(g, z)) < tol
        g = _df(lambda t, r: jnp.exp(-2 * (r * jnp.cos(t)) ** 2 - 2 * (r * jnp.sin(t)) ** 2))
        z = Chebfun.from_function(
            lambda x: 0.5 * jnp.exp(1j * jnp.pi * x), domain=Domain((-1.0, 1.0))
        )
        # g is radial: g = exp(-2 r^2), so on the r = 0.5 circle it is
        # exp(-0.5); the line integral is exp(-0.5) * (circumference = pi).
        ref = np.pi * np.exp(-0.5)
        assert abs(ref - float(g.integral(z))) < _TOL

    def test_line_integral_diameter(self):
        # pass(6): z = x exp(i pi/4); abs(sum(g(pi/4,:)) - integral(g, z)) < tol
        g = _df(lambda t, r: jnp.exp(-2 * (r * jnp.cos(t)) ** 2 - 2 * (r * jnp.sin(t)) ** 2))
        z = Chebfun.from_function(
            lambda x: x * jnp.exp(1j * jnp.pi / 4), domain=Domain((-1.0, 1.0))
        )
        # Diameter integral of the radial g: int_{-1}^{1} exp(-2 x^2) dx.
        ref = 2.0 * np.sqrt(np.pi / 8.0) * erf(np.sqrt(2.0))
        assert abs(ref - float(g.integral(z))) < _TOL
