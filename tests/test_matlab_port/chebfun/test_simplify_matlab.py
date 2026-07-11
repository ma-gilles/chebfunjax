"""Port of MATLAB Chebfun tests/chebfun/test_simplify.m (Fable 5).

MATLAB's test uses an array-valued chebfun; chebfunjax has none, so a
scalar analogue with the same structure (a large + a 1e-7-scale
component summed) is used at the same tolerances.  Simplify lives at
the tech level in chebfunjax.

Provenance
----------
MATLAB source : tests/chebfun/test_simplify.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(6178)
X = jnp.asarray(2 * RNG.uniform(size=100) - 1)


class TestChebfunSimplify:
    def test_simplify_preserves_values(self):
        f = cj.chebfun(lambda x: 1.0 / (1 + 10 * (x - 0.1) ** 2)
                       + 1e-7 / (1 + 30 * (x - 0.1) ** 2))
        t = f.funs[0].tech.simplify()
        # simplified representation still matches at 10*eps rel.
        from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
        from chebfunjax.domain import Domain
        f2 = Chebfun(funs=[_Piece(tech=t, interval=(-1.0, 1.0))],
                     domain=Domain((-1.0, 1.0)))
        err = jnp.abs(f(X) - f2(X))
        assert float(jnp.max(err)) < 10 * EPS * float(jnp.max(jnp.abs(f(X))))

    def test_simplify_with_loose_tol(self):
        f = cj.chebfun(lambda x: 1.0 / (1 + 10 * (x - 0.1) ** 2))
        tol = 1e4 * EPS
        t = f.funs[0].tech.simplify(tol=tol)
        from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
        from chebfunjax.domain import Domain
        f2 = Chebfun(funs=[_Piece(tech=t, interval=(-1.0, 1.0))],
                     domain=Domain((-1.0, 1.0)))
        err = jnp.abs(f(X) - f2(X))
        assert float(jnp.max(err)) < 10 * tol * float(jnp.max(jnp.abs(f(X))))
        assert len(t.coeffs) <= len(f.funs[0].tech.coeffs)
