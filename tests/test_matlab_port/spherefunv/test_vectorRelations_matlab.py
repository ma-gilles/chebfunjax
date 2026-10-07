"""All three original differential identities and bounds, without sampled norms.

Provenance
----------
MATLAB source : tests/spherefunv/test_vectorRelations.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df

Only the Cartesian callback is expressed in the Python spherical-coordinate API.
The global infinity norms and original 3000*chebfun2eps bound are unchanged.
This replaces the former wrong-expression/or-True clause; passing is not assumed.
"""
import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun


class TestSpherefunvVectorrelations:
    def test_all_matlab_assertions(self):
        tol = 3e3 * ChebfunPref().cheb2Prefs.chebfun2eps

        def cartesian_expression(lam, theta):
            x = jnp.cos(lam) * jnp.sin(theta)
            y = jnp.sin(lam) * jnp.sin(theta)
            z = jnp.cos(theta)
            return jnp.cos((x + .1) * y * z)

        f = Spherefun.from_function(cartesian_expression)
        # Source pass(1): norm(div(grad(f)) - laplacian(f), inf).
        assert (f.gradient().divergence() - f.laplacian()).norm(jnp.inf) < tol
        # Source pass(2): norm(div(curl(f)), inf), NOT vort(grad(f)).
        assert f.curl().divergence().norm(jnp.inf) < tol
        # Source pass(3): norm(vort(grad(f)), inf).
        assert f.gradient().vorticity().norm(jnp.inf) < tol
