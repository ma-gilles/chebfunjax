"""Port of MATLAB Chebfun tests/chebfun/test_tan.m (Fable 5).

The SingFun-wired factory builds tan on a domain broken at its poles, with
a simple pole ('exps' = -1) at every breakpoint.

Provenance
----------
MATLAB source : tests/chebfun/test_tan.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestChebfunTan:
    def test_poles_bug_568(self):
        # Bug from #568: tan(x) on pi*(-5/2 : 5/2), with a simple pole at each
        # of the six breakpoints ('exps' = -ones(1, 6), one shared exponent per
        # breakpoint).  The construction must stay short (length < 2^16).
        dom = tuple(np.pi * np.arange(-2.5, 3.0, 1.0))
        f = cj.chebfun(jnp.tan, domain=dom, exps=tuple([-1.0] * 6))

        # Every piece is a simple pole at both ends.
        from chebfunjax.fun.singfun import Singfun
        for p in f.funs:
            assert isinstance(p.tech, Singfun)
            assert p.tech.exponents == (-1.0, -1.0)

        # length(f) = total polynomial length across pieces < 2^16.
        total = sum(int(p.tech.smoothPart.n) for p in f.funs)
        assert total < 2 ** 16

        # Values away from the poles match tan.
        x = np.array([-2.0 * np.pi, -np.pi, 0.3, np.pi, 2.0 * np.pi])
        err = np.max(np.abs(np.asarray(f(jnp.asarray(x))) - np.tan(x)))
        assert err < 1e-12
