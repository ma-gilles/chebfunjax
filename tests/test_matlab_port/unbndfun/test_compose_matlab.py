"""Port of MATLAB Chebfun tests/unbndfun/test_compose.m (Opus 4.8).

``compose`` on an unbndfun composes an operator/second-fun with the underlying
Chebyshev representation on [-1, 1].  chebfunjax has no ``Unbndfun.compose``
method, but the exact building block exists on the onefun:

* ``compose(f, @op)``     ==  Unbndfun.from_chebtech(f.onefun.compose(op), dom)
* ``compose(f, @plus, g)`` ==  f + g  (binary operator)
* ``compose(f, g_bnd)``    ==  Unbndfun.from_chebtech(f.onefun.compose(g_bnd.onefun), dom)
  (function composition g(f); the range of f must lie inside [-1, 1]).

These are faithful because ``op(f(x)) = op(f.onefun(inv(x)))`` and the
reference variable is preserved through the wrapper.

Provenance
----------
MATLAB source : tests/unbndfun/test_compose.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.unbndfun import Unbndfun

EPS = float(np.finfo(np.float64).eps)
INF = np.inf
DOM = (0.0, INF)
X = jnp.asarray(np.linspace(0, 1e2, 100))


def _U(op, dom=DOM):
    return Unbndfun.from_function(op, Domain(dom))


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestUnbndfunCompose:
    def test_op_of_unbndfun(self):
        # compose(f, @sin) with f = exp(-x): sin(exp(-x))
        f = _U(lambda x: jnp.exp(-x))
        g = Unbndfun.from_chebtech(f.onefun.compose(jnp.sin), f.domain)
        gexact = jnp.sin(jnp.exp(-X))
        assert _ninf(g(X) - gexact) < 1e1 * EPS * g.vscale

    def test_plus_of_two_unbndfuns(self):
        # compose(f, @plus, g) == f + g, with (x+1)*exp(-x) as the result
        f = _U(lambda x: jnp.exp(-x))
        g = _U(lambda x: x * jnp.exp(-x))
        h = f + g
        hexact = (X + 1) * jnp.exp(-X)
        assert _ninf(h(X) - hexact) < 1e1 * EPS * h.vscale

    def test_bndfun_of_unbndfun(self):
        # compose(f, g) with g a bndfun cos on [-1,1]: cos(exp(-x))
        f = _U(lambda x: jnp.exp(-x))
        gb = Bndfun.from_function(jnp.cos, Domain((-1.0, 1.0)))
        h = Unbndfun.from_chebtech(f.onefun.compose(gb.onefun), f.domain)
        hexact = jnp.cos(jnp.exp(-X))
        assert _ninf(h(X) - hexact) < 1e1 * EPS * h.vscale
