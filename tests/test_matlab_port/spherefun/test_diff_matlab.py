"""Source-shaped port of MATLAB tests/spherefun/test_diff.m.

Provenance: Chebfun commit 7574c77. The ten predicates below retain the
native sample grid, infinity norm, and preference-derived tolerance. The last
pointwise identity is retained from the former port as a supplemental control.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.utils.quadrature import trigpts


def _sample_error(exact, approx, lam, th):
    expected = np.asarray(exact(*np.meshgrid(lam, th)))
    actual = np.asarray(approx.fevalm(lam, th))
    return np.linalg.norm((expected - actual).ravel(), ord=np.inf)


class TestSpherefunDiff:
    def test_all_ten_matlab_sample_error_clauses(self):
        tol = 2e2 * ChebfunPref().cheb2Prefs.chebfun2eps
        lam = trigpts(12, [-jnp.pi, jnp.pi])[0]
        th = jnp.linspace(0.0, jnp.pi, 6)
        x = lambda L, T: jnp.cos(L) * jnp.sin(T)
        y = lambda L, T: jnp.sin(L) * jnp.sin(T)
        z = lambda L, T: jnp.cos(T)
        errors = []

        # Native passes (1)-(3): Cartesian derivatives of x.
        f = Spherefun.from_function(x)
        errors.append(_sample_error(lambda L,T: 1-x(L,T)**2, f.diff(1), lam, th))
        errors.append(_sample_error(lambda L,T: -x(L,T)*y(L,T), f.diff(2), lam, th))
        errors.append(_sample_error(lambda L,T: -x(L,T)*z(L,T), f.diff(3), lam, th))

        # Native passes (4)-(6): Cartesian derivatives of z.
        f = Spherefun.from_function(z)
        errors.append(_sample_error(lambda L,T: -x(L,T)*z(L,T), f.diff(1), lam, th))
        errors.append(_sample_error(lambda L,T: -y(L,T)*z(L,T), f.diff(2), lam, th))
        errors.append(_sample_error(lambda L,T: 1-z(L,T)**2, f.diff(3), lam, th))

        # Native passes (7)-(9): shifted-y example, a=1/sqrt(2).
        a = 1 / jnp.sqrt(2.0)
        f = Spherefun.from_function(lambda L,T: jnp.sin(L-a)*jnp.sin(T))
        errors.append(_sample_error(
            lambda L,T: 0.25*(2*jnp.cos(L)*jnp.cos(2*T)*jnp.sin(L-a)
                              -jnp.sin(2*L-a)-3*jnp.sin(a)),
            f.diff(1), lam, th))
        errors.append(_sample_error(
            lambda L,T: 0.25*(2*jnp.sin(L)*jnp.cos(2*T)*jnp.sin(L-a)
                              +jnp.cos(2*L-a)+3*jnp.cos(a)),
            f.diff(2), lam, th))
        errors.append(_sample_error(
            lambda L,T: -0.5*jnp.sin(L-a)*jnp.sin(2*T),
            f.diff(3), lam, th))

        # Native pass (10): derivative of a nonseparable spherical Gaussian.
        lam0 = jnp.pi / jnp.sqrt(2.0)
        th0 = (jnp.sqrt(5.0)-2.0) / 2.0
        sig2 = 1.0
        def r2(L,T):
            return 2*(1-(jnp.sin(T)*jnp.sin(th0))*jnp.cos(L-lam0)
                      -jnp.cos(T)*jnp.cos(th0))
        f = Spherefun.from_function(lambda L,T: jnp.exp(-sig2*r2(L,T)))
        def gaussian_dx(L,T):
            return -0.5*sig2*jnp.exp(-sig2*r2(L,T))*(
                2*jnp.cos(L)*jnp.cos(th0)*jnp.sin(2*T)
                +(jnp.cos(2*L-lam0)-3*jnp.cos(lam0)
                  -2*jnp.cos(L)*jnp.cos(L-lam0)*jnp.cos(2*T))*jnp.sin(th0))
        errors.append(_sample_error(gaussian_dx, f.diff(1), lam, th))

        assert len(errors) == 10
        assert max(errors) < tol, f"native pass errors={errors}, tol={tol}"

    def test_supplemental_pointwise_z_derivative_control(self):
        fz = Spherefun.from_function(lambda lam, th: jnp.cos(th))
        dz = fz.diff(dim=3)
        lam, th = jnp.asarray(0.6), jnp.asarray(1.1)
        z = jnp.cos(th)
        assert abs(float(dz(lam, th)) - (1-float(z)**2)) < 1e-9
