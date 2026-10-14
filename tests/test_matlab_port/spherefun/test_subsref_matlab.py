"""The twelve original assertions in MATLAB tests/spherefun/test_subsref.m."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.chebfun2d._svd import _as_fun
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

from ._cart import sph_xyz

jax.config.update("jax_enable_x64", True)


def _factor_chebfun(factor):
    # Preserve the native factor coefficients and interpret their reference
    # interval as the source physical trig domain.
    return _as_fun(factor, (-np.pi, np.pi))


def test_source_subsref_assertions_one_through_ten():
    tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps

    # pass(1): native Cartesian call at the literal source point.
    ff = lambda x, y, z: z*jnp.sin(x*(y-0.1))
    f = sph_xyz(ff)
    x = y = 1/jnp.sqrt(3.0)
    z = -1/jnp.sqrt(3.0)
    assert abs(float(f(x, y, z) - ff(x, y, z))) < tol

    # pass(2): angular call at the literal source point.
    ff2 = lambda lam, th: jnp.exp(jnp.cos(lam)*jnp.sin(th)*jnp.cos(th))
    f2 = Spherefun.from_function(ff2)
    lam, th = np.pi/3, np.pi/4
    assert abs(float(f2(lam, th) - ff2(lam, th))) < tol

    # pass(3): both theta slices, compared with the source trig Chebfuns.
    for th0 in (0.2, 0.7):
        got = f2.slice_theta(th0)
        want = chebfun(lambda t, th0=th0: ff2(t, th0),
                       domain=(-np.pi, np.pi), trig=True)
        assert float((got-want).norm()) < tol

    # pass(4): both lambda slices, compared with source trig Chebfuns.
    for lam0 in (0.2, 0.7):
        got = f2.slice_lambda(lam0)
        want = chebfun(lambda t, lam0=lam0: ff2(lam0, t),
                       domain=(-np.pi, np.pi), trig=True)
        assert float((got-want).norm()) < tol

    # pass(5): the source constant-z circle.
    ff3 = lambda x, y, z: z*jnp.sin(x*(y-0.1))
    f3 = sph_xyz(ff3)
    z0 = 0.1
    th0 = np.arccos(z0)
    got = f3.slice_z(z0)
    want = chebfun(
        lambda lam: ff3(jnp.cos(lam)*jnp.sin(th0),
                       jnp.sin(lam)*jnp.sin(th0), z0),
        domain=(-np.pi, np.pi), trig=True)
    assert float((got-want).norm()) < tol

    # pass(6): source x=constant circle uses y=r*cos(t), z=r*sin(t).
    x0 = 0.1
    got = f3.slice_x(x0)
    radius = jnp.sqrt(1-x0*x0)
    want = chebfun(lambda t: ff3(x0, radius*jnp.cos(t), radius*jnp.sin(t)),
                   domain=(-np.pi, np.pi), trig=True)
    assert float((got-want).norm()) < tol

    # pass(7): source y=constant circle uses x=r*cos(t), z=r*sin(t).
    y0 = -0.4
    got = f3.slice_y(y0)
    radius = jnp.sqrt(1-y0*y0)
    want = chebfun(lambda t: ff3(radius*jnp.cos(t), y0, radius*jnp.sin(t)),
                   domain=(-np.pi, np.pi), trig=True)
    assert float((got-want).norm()) < tol

    # pass(8)-(9): source compares raw factor coefficients on [-pi,pi].
    f4 = Spherefun.from_function(
        lambda lam, th: jnp.cos(lam)*jnp.sin(th))
    row = _factor_chebfun(f4.rows[0])
    col = _factor_chebfun(f4.cols[0])
    row_ref = chebfun(jnp.cos, domain=(-np.pi, np.pi), trig=True)
    col_ref = chebfun(jnp.sin, domain=(-np.pi, np.pi), trig=True)
    row_error = min(float((row-row_ref).norm()), float((row+row_ref).norm()))
    col_error = min(float((col-col_ref).norm()), float((col+col_ref).norm()))
    assert row_error < tol
    assert col_error < tol

    # pass(10): physical domain accessor.
    assert float(jnp.linalg.norm(
        f4.domain-jnp.asarray([-np.pi, np.pi, 0.0, np.pi]))) < tol


def test_source_subsref_composition_assertions_eleven_and_twelve():
    tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
    ff = lambda x, y, z: z + jnp.sin(jnp.pi*x*y)
    f = sph_xyz(ff)
    g = chebfun(lambda t: t**2, domain=(-1.5, 1.5))
    h_true = sph_xyz(lambda x, y, z: (z+jnp.sin(jnp.pi*x*y))**2)

    # pass(11): scalar Chebfun composition G(f).
    h = g(f)
    assert float((h-h_true).norm()) < tol

    # pass(12): three-column Chebfun composition returns a Spherefunv.
    G = chebfun(lambda t: jnp.stack((t**2, t, -t**2), axis=-1),
                domain=(-1.5, 1.5))
    H_true = Spherefunv(
        sph_xyz(lambda x, y, z: (z+jnp.sin(jnp.pi*x*y))**2),
        sph_xyz(ff),
        sph_xyz(lambda x, y, z: -(z+jnp.sin(jnp.pi*x*y))**2))
    H = G(f)
    assert isinstance(H, Spherefunv)
    assert float((H-H_true).norm()) < tol


def test_spherefun_chebfun_compose_empty_order_source_control():
    empty_sphere = Spherefun.empty()
    empty_cheb = Chebfun.empty()
    # @spherefun/compose.m: empty operator returns f first; otherwise an
    # empty f is replaced by the operator. These are source controls, not
    # additional entries in the native test_subsref pass vector.
    nonempty_cheb = chebfun(lambda t: t, domain=(-1.0, 1.0))
    assert not nonempty_cheb.isempty()
    empty_op_result = empty_sphere.compose(empty_cheb)
    assert isinstance(empty_op_result, Spherefun)
    assert empty_op_result.isempty()
    empty_f_result = empty_sphere.compose(nonempty_cheb)
    assert isinstance(empty_f_result, Chebfun)
    assert not empty_f_result.isempty()
    assert float(empty_f_result(0.25)) == float(nonempty_cheb(0.25))


def test_empty_sphere_compose_callable_source_control():
    def op(value):
        return value**2
    assert Spherefun.empty().compose(op) is op
