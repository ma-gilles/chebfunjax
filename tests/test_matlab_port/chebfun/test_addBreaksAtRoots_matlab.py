"""Port of MATLAB Chebfun tests/chebfun/test_addBreaksAtRoots.m
(Fable 5).

FIXED: addBreaksAtRoots/addBreaks added in the Fable 5 audit.  The
MATLAB pointValues(2) == 0 assertion is carried as "the function
value at the new breakpoint is 0" (chebfunjax has no pointValues
storage); singfun/exps cases skipped.

Provenance
----------
MATLAB source : tests/chebfun/test_addBreaksAtRoots.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestChebfunAddBreaksAtRoots:
    def test_scalar(self):
        f = cj.chebfun(lambda x: jnp.sin(x) - 0.5)
        g = f.addBreaksAtRoots()
        breaks = [float(p.interval[0]) for p in g.funs]
        root = float(np.arcsin(0.5))
        # a break lands at the root, and the value there is 0
        assert min(abs(b - root) for b in breaks) < 1e-13
        assert abs(float(g(jnp.asarray(root)))) < 1e-14
        # values unchanged elsewhere
        xs = jnp.asarray(np.linspace(-0.95, 0.95, 50))
        assert float(jnp.max(jnp.abs(g(xs) - f(xs)))) < 1e-14

    def test_no_interior_roots(self):
        f = cj.chebfun(lambda x: 2 + jnp.sin(x))
        g = f.addBreaksAtRoots()
        assert len(g.funs) == len(f.funs)
