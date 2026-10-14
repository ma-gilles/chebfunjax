"""Source-shaped port of MATLAB tests/spherefun/test_grad.m.

The seven active predicates retain source examples and preference-based
bounds. The prior Dirichlet-energy check is a separate supplemental control.

Provenance
----------
MATLAB source : tests/spherefun/test_grad.m
Chebfun commit: 7574c77
"""
from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv


def _xyz_fun(fc):
    def f(lam, th):
        x = jnp.cos(lam) * jnp.sin(th)
        y = jnp.sin(lam) * jnp.sin(th)
        z = jnp.cos(th)
        return fc(x, y, z)
    return Spherefun.from_function(f)


def _exact_vector(fx, fy, fz):
    return Spherefunv(_xyz_fun(fx), _xyz_fun(fy), _xyz_fun(fz))


class TestSpherefunGrad:
    def test_all_seven_active_matlab_assertions(self):
        tol = 1e2 * ChebfunPref().cheb2Prefs.chebfun2eps

        # Native pass(1): empty input returns an empty vector spherefun.
        empty_grad = Spherefun.empty().grad()
        assert isinstance(empty_grad, Spherefunv) and empty_grad.isempty()

        # Native pass(2): gradient of zero has zero vector norm.
        zero = _xyz_fun(lambda x,y,z: 0*x)
        assert float(zero.grad().norm()) < tol

        # Native pass(3): grad((x-y)z) equals the explicit Cartesian field.
        f = _xyz_fun(lambda x,y,z: (x-y)*z)
        exact = _exact_vector(
            lambda x,y,z: (1+2*x*(y-x))*z,
            lambda x,y,z: (-1+2*y*(y-x))*z,
            lambda x,y,z: -(x-y)*(2*z**2-1))
        assert float((f.grad()-exact).norm()) < tol

        # Native pass(4): grad(cos(4z)).
        f = _xyz_fun(lambda x,y,z: jnp.cos(4*z))
        exact = _exact_vector(
            lambda x,y,z: 4*x*z*jnp.sin(4*z),
            lambda x,y,z: 4*y*z*jnp.sin(4*z),
            lambda x,y,z: -4*(1-z**2)*jnp.sin(4*z))
        assert float((f.grad()-exact).norm()) < tol

        # Native pass(5): grad(cos(4x)), with the source 20x multiplier.
        f = _xyz_fun(lambda x,y,z: jnp.cos(4*x))
        exact = _exact_vector(
            lambda x,y,z: 4*(x**2-1)*jnp.sin(4*x),
            lambda x,y,z: 4*x*y*jnp.sin(4*x),
            lambda x,y,z: 4*x*z*jnp.sin(4*x))
        assert float((f.grad()-exact).norm()) < 20*tol

        # Native pass(6): grad(cos(4y)), with the source 10x multiplier.
        f = _xyz_fun(lambda x,y,z: jnp.cos(4*y))
        exact = _exact_vector(
            lambda x,y,z: 4*x*y*jnp.sin(4*y),
            lambda x,y,z: 4*(y**2-1)*jnp.sin(4*y),
            lambda x,y,z: 4*y*z*jnp.sin(4*y))
        u = f.grad()
        assert float((u-exact).norm()) < 10*tol

        # Native pass(7): grad and gradient return the same Spherefunv.
        grad_field = f.grad()
        gradient_field = f.gradient()
        assert isinstance(grad_field, Spherefunv)
        assert all(a.isequal(b) for a,b in
                   zip(grad_field.components, gradient_field.components))

    def test_supplemental_grad_component_iteration_compatibility(self):
        f = Spherefun.sphharm(2, 1)
        g = f.grad()
        assert len(g) == 3
        gx, gy, gz = g
        assert all(isinstance(component, Spherefun) for component in (gx, gy, gz))
        empty = Spherefun.empty().grad()
        assert isinstance(empty, Spherefunv) and empty.isempty()
        assert len(empty) == 0 and tuple(empty) == ()

    def test_supplemental_dirichlet_energy_of_harmonic(self):
        l, m = 3, 2
        Y = Spherefun.sphharm(l, m)
        gx, gy, gz = Y.grad()
        e = Spherefun.from_function(
            lambda lam, th: gx(lam, th)**2 + gy(lam, th)**2 + gz(lam, th)**2)
        assert abs(float(e.sum2()) - l*(l+1)) < 1e-4
