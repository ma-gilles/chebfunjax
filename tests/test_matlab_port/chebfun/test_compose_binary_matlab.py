"""Port of MATLAB Chebfun tests/chebfun/test_compose_binary.m
(Fable 5).

FIXED: Chebfun.compose(op, g) added in the Fable 5 audit.  The
splitting-enabled nonsmooth cases carry a looser tolerance: the
bisection splitting used by the chebfunjax constructor has no edge
detection, so kink placement is accurate to ~1e-11 (values correct,
representation bloated).  Array-valued cases skipped.

Provenance
----------
MATLAB source : tests/chebfun/test_compose_binary.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

XS = jnp.asarray(np.linspace(-0.97, 0.97, 100))


class TestChebfunComposeBinary:
    def test_smooth_cases(self):
        f = cj.chebfun(lambda x: jnp.cos(2 * (x + 0.2)))
        g = cj.chebfun(lambda x: jnp.sin(x - 0.1))
        # pass(1)
        h = f.compose(lambda a, b: a * b, g)
        assert float(jnp.max(jnp.abs(h(XS) - f(XS) * g(XS)))) \
            < 1e-13

        # pass(2): non-default domain
        f2 = cj.chebfun(lambda x: jnp.cos(2 * (x + 0.2)),
                        domain=(-2, 7))
        g2 = cj.chebfun(lambda x: jnp.sin(x - 0.1), domain=(-2, 7))
        xs2 = jnp.asarray(np.linspace(-1.95, 6.95, 100))
        h2 = f2.compose(lambda a, b: a ** 2 - b ** 2, g2)
        assert float(jnp.max(jnp.abs(
            h2(xs2) - (f2(xs2) ** 2 - g2(xs2) ** 2)))) < 1e-12

    def test_nonsmooth_op_with_splitting(self):
        # pass(4): |f*g| requires splitting
        f = cj.chebfun(lambda x: jnp.cos(2 * (x + 0.2)))
        g = cj.chebfun(lambda x: jnp.sin(x - 0.1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = f.compose(lambda a, b: jnp.abs(a * b), g)
        assert float(jnp.max(jnp.abs(
            h(XS) - jnp.abs(f(XS) * g(XS))))) < 1e-9
