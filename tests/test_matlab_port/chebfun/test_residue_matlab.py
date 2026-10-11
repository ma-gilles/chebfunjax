"""Original residue predicates, Chebfun 7574c77.

Provenance
----------
MATLAB source: tests/chebfun/test_residue.m.
Chebfun commit: 7574c77.
"""
import jax.numpy as jnp

import chebfunjax as cj


class TestChebfunResidue:
    def test_all_matlab_assertions(self):
        f = cj.chebfun(lambda x: (x-1.1)*(x**2+1)*(x-10j))
        g = cj.chebfun(lambda x: x**5)
        r, p, k = cj.residue(g, f)
        G, F = cj.residue(r, p, k)
        tol = 100*jnp.finfo(jnp.float64).eps
        pexact = jnp.asarray([10j, 1.1, -1j, 1j])
        assert (jnp.linalg.norm(jnp.sort(p.real)-jnp.sort(pexact.real))
                + jnp.linalg.norm(jnp.sort(p.imag)-jnp.sort(pexact.imag))) < 10*tol
        product = g*F
        assert (product-G*f).normest()/product.vscale < 10*tol*k.vscale
        assert (k-cj.chebfun(lambda x: x+10j+1.1)).normest() < 10*tol
        B, A = cj.residue(jnp.asarray([1.,1.]), jnp.asarray([1.,-1.]), cj.Chebfun.empty())
        assert (B-cj.chebfun(lambda x: 2*x)).normest() < tol
        assert (A-cj.chebfun(lambda x: x**2-1)).normest() < tol
