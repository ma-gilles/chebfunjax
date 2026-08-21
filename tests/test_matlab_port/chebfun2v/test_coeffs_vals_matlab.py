"""Port of MATLAB Chebfun tests/chebfun2v/test_coeffs_vals.m (Fable 5).

MATLAB's randnfun2 samples don't reproduce across RNGs; smooth
deterministic fields exercise the same identities.

Provenance
----------
MATLAB source : tests/chebfun2v/test_coeffs_vals.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

jax.config.update("jax_enable_x64", True)

TOL = 1e3 * 2.220446049250313e-16


def _dev(f2, g2):
    xs = jnp.linspace(-0.9, 0.9, 9)
    X, Y = jnp.meshgrid(xs, xs)
    return float(jnp.max(jnp.abs(jnp.asarray(f2(X, Y))
                                 - jnp.asarray(g2(X, Y)))))


class TestChebfun2vCoeffsVals:
    def test_all_matlab_assertions(self):
        u = Chebfun2.from_function(
            lambda x, y: jnp.cos(3 * x) * jnp.sin(2 * y) + x * y)
        v = Chebfun2.from_function(
            lambda x, y: jnp.exp(x) * jnp.cos(y ** 2))
        f = Chebfun2v([u.approx, v.approx])

        # coeffs2 roundtrip through chebfun2(x, 'coeffs').  pass(1)/(2)
        x_c, y_c = f.coeffs2()
        assert _dev(Chebfun2.from_coeffs(x_c), u) < TOL
        assert _dev(Chebfun2.from_coeffs(y_c), v) < TOL

        # fixed-size coeffs2 agree with the component call.  pass(3)/(4)
        x50, y50 = f.coeffs2(50, 60)
        assert float(jnp.max(jnp.abs(x50 - u.coeffs2(50, 60)))) < TOL
        assert float(jnp.max(jnp.abs(y50 - v.coeffs2(50, 60)))) < TOL

        # coeffs2chebfun2v.  pass(5)
        f2 = Chebfun2v.coeffs2chebfun2v(u.coeffs2(), v.coeffs2())
        for c_new, c_ref in zip(f2.components, f.components):
            assert _dev(Chebfun2(approx=c_new),
                        Chebfun2(approx=c_ref)) < TOL

        # coeffs2vals / vals2coeffs identities.  pass(6)-(9)
        x_c2, y_c2 = f2.coeffs2()
        uu, vv = Chebfun2v.coeffs2vals(x_c2, y_c2)
        assert float(jnp.max(jnp.abs(
            Chebfun2.coeffs2vals(x_c2) - uu))) < TOL
        assert float(jnp.max(jnp.abs(
            Chebfun2.coeffs2vals(y_c2) - vv))) < TOL
        a, b = Chebfun2v.vals2coeffs(uu, vv)
        assert float(jnp.max(jnp.abs(x_c2 - a))) < TOL
        assert float(jnp.max(jnp.abs(y_c2 - b))) < TOL
