"""Port of MATLAB Chebfun @chebfun/gamma.m documented behavior (Fable 5).

There is no dedicated tests/chebfun/test_gamma.m in the reference; this
mirrors the behaviour documented in @chebfun/gamma.m, whose example
builds the gamma function on [0.1, 3] by composing gamma with the
identity chebfun.  gamma(F) = compose(F, @gamma); it does not introduce
poles, so the range of F must avoid the non-positive integers.

Provenance
----------
MATLAB source : @chebfun/gamma.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.special import gamma as sgamma

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunGamma:
    def test_gamma_on_documented_interval(self):
        # The @chebfun/gamma.m example: x = chebfun('x', [0.1, 3]); gamma(x).
        x = cj.chebfun(lambda t: t, domain=[0.1, 3.0])
        g = x.gamma()
        xx = np.linspace(0.1, 3.0, 100)
        err = np.max(np.abs(np.asarray(g(jnp.asarray(xx))) - sgamma(xx)))
        assert err < 1e3 * EPS * float(g.vscale)

    def test_gamma_matches_direct_construction(self):
        # gamma(x) equals a direct adaptive chebfun of the gamma function.
        x = cj.chebfun(lambda t: t, domain=[1.0, 4.0])
        g = x.gamma()
        direct = cj.chebfun(lambda t: jnp.asarray(sgamma(np.asarray(t))),
                            domain=[1.0, 4.0])
        xx = np.linspace(1.0, 4.0, 80)
        err = np.max(np.abs(np.asarray(g(jnp.asarray(xx)))
                            - np.asarray(direct(jnp.asarray(xx)))))
        assert err < 1e3 * EPS * float(g.vscale)
