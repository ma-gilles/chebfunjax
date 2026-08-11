"""Port of MATLAB Chebfun tests/unbndfun/test_changeMap.m (Opus 4.8).

``changeMap(f, newDom)`` keeps the underlying onefun (the Chebyshev
representation on [-1, 1]) and only swaps the nonlinear algebraic map so it
targets a new unbounded interval.  chebfunjax does not expose ``changeMap``,
but the exact equivalent is ``Unbndfun.from_chebtech(f.onefun, Domain(newDom))``
-- the onefun is unchanged, only the domain (hence the map) differs.  For a
translation of the finite endpoint (a -> a', or b -> b') the reference point
is preserved: ``inv_new(xNew) == inv_old(x)``, so ``g(xNew) == f(x)``.

Provenance
----------
MATLAB source : tests/unbndfun/test_changeMap.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.domain import Domain
from chebfunjax.fun.unbndfun import Unbndfun

EPS = float(np.finfo(np.float64).eps)
INF = np.inf


def _U(op, dom):
    return Unbndfun.from_function(op, Domain(dom))


def _change_map(f, new_dom):
    """Faithful equivalent of unbndfun/changeMap: keep onefun, swap the map."""
    return Unbndfun.from_chebtech(f.onefun, Domain(new_dom))


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestUnbndfunChangeMap:
    # --- Functions on [a inf]: [1, inf) -> [7, inf) -------------------
    def test_decaying_right_inf(self):
        dom, dom_new = (1.0, INF), (7.0, INF)
        x = np.linspace(1, 1e2, 100)
        x_new = dom_new[0] + x - dom[0]
        op = lambda t: (1 - jnp.exp(-t)) / t
        f = _U(op, dom)
        g = _change_map(f, dom_new)
        assert _ninf(f(jnp.asarray(x)) - g(jnp.asarray(x_new))) < 1e1 * EPS * f.vscale

    def test_blowup_right_inf(self):
        # MATLAB uses exponents [0 1] for the linear-growth blowup; chebfunjax
        # cannot represent that, but changeMap only copies the onefun and swaps
        # the map, so the invariance f(x) == g(xNew) still holds exactly for
        # whatever (here unresolved) representation the constructor produced.
        dom, dom_new = (1.0, INF), (7.0, INF)
        x = np.linspace(1, 1e2, 100)
        x_new = dom_new[0] + x - dom[0]
        op = lambda t: t * (5 + jnp.exp(-t ** 3))
        f = _U(op, dom)
        g = _change_map(f, dom_new)
        assert _ninf(f(jnp.asarray(x)) - g(jnp.asarray(x_new))) < 1e1 * EPS * f.vscale

    # --- Functions on [-inf b]: [-inf, -3pi) -> [-inf, 200) -----------
    def test_x_exp_left_inf(self):
        dom, dom_new = (-INF, -3 * np.pi), (-INF, 200.0)
        x = np.linspace(-1e6, -3 * np.pi, 100)
        x_new = dom_new[1] + x - dom[1]
        op = lambda t: t * jnp.exp(t)
        f = _U(op, dom)
        g = _change_map(f, dom_new)
        assert _ninf(f(jnp.asarray(x)) - g(jnp.asarray(x_new))) < EPS * f.vscale

    def test_blowup_left_inf(self):
        # MATLAB uses exponents [0 -1] (pole at the finite endpoint) —
        # now representable as a Singfun onefun; sample interior points
        # (the pole itself evaluates to inf on both sides).
        dom, dom_new = (-INF, -3 * np.pi), (-INF, 200.0)
        b = dom[1]
        x = np.linspace(-1e6, -3 * np.pi, 100, endpoint=False)
        x_new = dom_new[1] + x - dom[1]
        op = lambda t: t * (5 + jnp.exp(t ** 3)) / (b - t)
        f = _U(op, dom)
        g = _change_map(f, dom_new)
        assert _ninf(f(jnp.asarray(x)) - g(jnp.asarray(x_new))) < EPS * f.vscale
