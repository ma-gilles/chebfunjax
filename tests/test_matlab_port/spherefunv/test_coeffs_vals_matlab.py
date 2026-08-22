"""Port of MATLAB Chebfun tests/spherefunv/test_coeffs_vals.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefunv/test_coeffs_vals.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

jax.config.update("jax_enable_x64", True)

TOL = 1e3 * 2.220446049250313e-16


def _sph(fc):
    def f(lam, th):
        x = jnp.cos(lam) * jnp.sin(th)
        y = jnp.sin(lam) * jnp.sin(th)
        z = jnp.cos(th)
        return fc(x, y, z)
    return Spherefun.from_function(f)


def _norm_inf(g, n=25):
    lam = jnp.linspace(-np.pi + 1e-6, np.pi - 1e-6, n)
    th = jnp.linspace(1e-3, np.pi - 1e-3, n)
    L, T = jnp.meshgrid(lam, th)
    return float(jnp.max(jnp.abs(jnp.asarray(g(L, T)))))


def _vnorm(F, n=25):
    return max(_norm_inf(c, n) for c in F.components)




class TestSpherefunvCoeffsVals:
    def test_all_matlab_assertions(self):
        # Deterministic smooth fields stand in for randnfunsphere.
        u = _sph(lambda x, y, z: jnp.cos(2 * x * y) + z)
        v = _sph(lambda x, y, z: jnp.sin(x + y * z))
        w = _sph(lambda x, y, z: jnp.exp(z) * jnp.cos(x))
        f = Spherefunv(u, v, w)

        # coeffs2 roundtrip through coeffs2spherefun.  pass(1)-(3)
        x_c, y_c, z_c = f.coeffs2()
        from chebfunjax.spherefun.spherefun import Spherefun
        for X, ref in ((x_c, u), (y_c, v), (z_c, w)):
            g = Spherefun.coeffs2spherefun(X)
            assert _norm_inf(g - ref) < 10 * TOL

        # fixed-size coeffs2 agree per component.  pass(4)-(6)
        x50, y50, z50 = f.coeffs2(50, 60)
        assert float(jnp.max(jnp.abs(
            x50 - u.coeffs2(50, 60)))) < TOL
        assert float(jnp.max(jnp.abs(
            y50 - v.coeffs2(50, 60)))) < TOL
        assert float(jnp.max(jnp.abs(
            z50 - w.coeffs2(50, 60)))) < TOL

        # coeffs2spherefunv.  pass(7)
        f2 = Spherefunv.coeffs2spherefunv(u.coeffs2(), v.coeffs2(),
                                          w.coeffs2())
        for c_new, ref in zip(f2.components, (u, v, w)):
            assert _norm_inf(c_new - ref) < 10 * TOL

        # coeffs2vals / vals2coeffs identities.  pass(8)-(13)
        x2, y2, z2 = f2.coeffs2()
        uu, vv, ww = Spherefunv.coeffs2vals(x2, y2, z2)
        assert float(jnp.max(jnp.abs(
            Spherefun.coeffs2vals(x2) - uu))) < TOL
        assert float(jnp.max(jnp.abs(
            Spherefun.coeffs2vals(y2) - vv))) < TOL
        a, b, c = Spherefunv.vals2coeffs(uu, vv, ww)
        for got, want in ((a, x2), (b, y2), (c, z2)):
            assert float(jnp.max(jnp.abs(
                got - jnp.asarray(want,
                                  dtype=jnp.complex128)))) < TOL
