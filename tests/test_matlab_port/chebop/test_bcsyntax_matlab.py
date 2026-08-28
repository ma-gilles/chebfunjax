"""Port of MATLAB Chebfun tests/chebop/test_bcsyntax.m (Fable 5).

A syntax smoke: the various boundary-condition spellings must be
accepted without error (MATLAB sets pass = true unconditionally).  A
solve sanity-check is added for the keyword forms.

Provenance
----------
MATLAB source : tests/chebop/test_bcsyntax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)


class TestChebopBcSyntax:
    def test_all_matlab_assertions(self):
        N = Chebop(lambda u: u.diff(2) - u.exp())
        N.lbc = 1.0
        N.rbc = lambda u: u.diff() - 2.0

        N = Chebop(lambda u: u.diff(2) - u.exp())
        N.bc = lambda u: [u(0.0) - 1.0, u.sum()]

        N = Chebop(lambda u: u.diff(3) - u.exp())
        N.lbc = "dirichlet"
        N.rbc = [1.0, "neumann"]
        N.bc = lambda u: u(0.0) - 1.0

        # Solve sanity: keyword conditions on a linear problem.
        L = Chebop(lambda u: u.diff(2) + u, domain=(0.0, 1.0))
        L.lbc = "dirichlet"      # u(0) = 0
        L.rbc = "neumann"        # u'(1) = 0
        u = L.solve(1.0)
        assert abs(float(u(jnp.asarray(0.0)))) < 1e-8
        assert abs(float(u.diff()(jnp.asarray(1.0)))) < 1e-7
