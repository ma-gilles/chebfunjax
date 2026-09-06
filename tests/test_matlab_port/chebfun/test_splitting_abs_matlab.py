"""Port of MATLAB Chebfun tests/chebfun/test_splitting_abs.m (Fable 5).

The final MATLAB case (``'splitting', 1`` instead of ``'on'`` must be
rejected by the flag parser) has no counterpart in a keyword API and is
not asserted.

Provenance
----------
MATLAB source : tests/chebfun/test_splitting_abs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)

EPS = np.finfo(float).eps


class TestChebfunSplittingAbs:
    def test_all_matlab_assertions(self):
        cheps = ChebfunPref().chebfuneps
        rng = np.random.RandomState(6178)
        FF = [jnp.abs, lambda x: jnp.abs(x) ** 5, lambda x: jnp.abs(jnp.sin(10 * x)),
              lambda x: jnp.abs(jnp.sin(30 * x))]
        xx = jnp.asarray(np.linspace(-1, 1, 100))
        for F in FF:
            f = chebfun(F, domain=(-1.0, 1.0), splitting=True)
            err = float(np.max(np.abs(np.asarray(f(xx)) - np.asarray(F(xx)))))
            assert err < 50 * EPS
            assert err < 1000 * cheps

        dom = (-1.0, 1.0)
        dom_check = (dom[0] + 0.1, dom[1] - 0.1)
        pow1, pow2 = -0.5, -0.2
        op = lambda x: jnp.abs(jnp.cos(25 * x) * (x - dom[0]) ** pow1  # noqa: E731
                               * (dom[1] - x) ** pow2)
        f = chebfun(op, domain=dom, exps=(pow1, pow2), splitting=True)
        x = jnp.asarray((dom_check[1] - dom_check[0]) * rng.rand(100) + dom_check[0])
        vals_f = np.asarray(f(x))
        vals_check = np.asarray(op(x))
        err = vals_f - vals_check
        assert np.max(np.abs(err - np.mean(err))) < 1e6 * EPS * np.max(np.abs(vals_check))

        dom = (0.0, jnp.inf)
        dom_check = (0.0, 100.0)
        x = jnp.asarray((dom_check[1] - dom_check[0]) * rng.rand(100) + dom_check[0])
        op = lambda x: 0.75 + jnp.sin(10 * x) / jnp.exp(x)  # noqa: E731
        f = chebfun(op, domain=dom, splitting=True)
        g = abs(f)
        err = np.asarray(g(x)) - np.abs(np.asarray(op(x)))
        assert np.max(np.abs(err)) < 1e6 * EPS * g.vscale
