"""Port of MATLAB Chebfun tests/singfun/test_compose.m (Opus 4.8).

chebfunjax Singfun implements no ``compose`` method, so every assertion is
xfailed (the call raises ``AttributeError``).

Provenance
----------
MATLAB source : tests/singfun/test_compose.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech2

EPS = float(np.finfo(np.float64).eps)

X = jnp.asarray(np.linspace(-0.99, 0.99, 100))
_REASON = "chebfunjax Singfun has no compose() method"


def _sf(f, exps):
    return Singfun.from_function(f, exps)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestSingfunCompose:
    def test_compose_two_singfuns(self):
        f = _sf(lambda x: jnp.sin(x) / (x + 1), (-1.0, 0.0))
        g = _sf(lambda x: jnp.cos(x) / (x + 1), (-1.0, 0.0))
        h = f.compose(np.add, g)
        exact = (jnp.sin(X) + jnp.cos(X)) / (X + 1)
        assert _ninf(h(X) - exact) < 1e3 * EPS * h.smoothPart.vscale

    def test_compose_operator_and_singfun(self):
        f = _sf(lambda x: jnp.sqrt(x + 1), (0.5, 0.0))
        h = f.compose(jnp.sin)
        exact = jnp.sin(jnp.sqrt(X + 1))
        assert _ninf(h(X) - exact) < 1e1 * EPS * h.smoothPart.vscale

    def test_compose_smoothfun_and_singfun(self):
        f = _sf(lambda x: jnp.sqrt(x + 1), (0.5, 0.0))
        g = Chebtech2.from_function(lambda x: jnp.cos(x))
        h = f.compose(g)
        exact = jnp.cos(jnp.sqrt(X + 1))
        assert _ninf(h(X) - exact) < 1e4 * EPS * h.smoothPart.vscale
