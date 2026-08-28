"""Port of MATLAB Chebfun tests/chebop/test_domain.m (Fable 5).

MATLAB's column-vector-domain error identifiers have no Python analog;
the invalid-domain passes map to rejecting malformed domains.

Provenance
----------
MATLAB source : tests/chebop/test_domain.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)


def _n(f, d):
    xs = jnp.linspace(d[0] + 1e-9, d[1] - 1e-9, 33)
    return float(jnp.max(jnp.abs(jnp.asarray(f(xs)))))


class TestChebopDomain:
    def test_all_matlab_assertions(self):
        dom = (0.0, 2.0)
        # pass(1)/(2): malformed domains are rejected.
        try:
            Chebop(lambda u: u.diff(2) + u, domain=(2.0,))
            raised = False
        except Exception:
            raised = True
        assert raised

        # pass(3): setting N.domain after construction matches passing
        # the domain to the constructor.
        N = Chebop(lambda u: u.diff(2) + u, domain=dom)
        N.bc = 0.0
        u1 = N.solve(1.0)
        N = Chebop(lambda u: u.diff(2) + u)
        N.domain = dom
        N.bc = 0.0
        u2 = N.solve(1.0)
        assert _n(u1 - u2, dom) < 1e-12
