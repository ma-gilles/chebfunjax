"""Port of MATLAB Chebfun tests/chebfun2v/test_syntax.m (Fable 5).

MATLAB's four constructor spellings (handles + domain, cell of handles,
chebfun2 objects, objects + domain) map to from_functions and the
components-list constructor; all must agree.

Provenance
----------
MATLAB source : tests/chebfun2v/test_syntax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

TOL = 1e5 * float(np.finfo(np.float64).eps)
DOMS = [(-1.0, 1.0, -1.0, 1.0), (-1.0, 1.0, 1.0, 2.0)]


def _maxdiff(F, G, dom):
    xs = np.linspace(dom[0] + 0.05, dom[1] - 0.05, 4)
    ys = np.linspace(dom[2] + 0.05, dom[3] - 0.05, 4)
    out = 0.0
    for a, b in zip(F.components, G.components):
        fa, fb = Chebfun2(approx=a), Chebfun2(approx=b)
        out = max(out, max(
            abs(float(np.asarray(fa(x, y))) - float(np.asarray(fb(x, y))))
            for x in xs for y in ys))
    return out


class TestChebfun2vSyntax:
    @pytest.mark.parametrize("dom", DOMS)
    def test_constructor_spellings_agree(self, dom):
        f = lambda x, y: jnp.cos(x) + jnp.sin(x * y)  # noqa: E731
        g = lambda x, y: jnp.cos(x * y)  # noqa: E731
        fcheb = Chebfun2.from_function(f, domain=dom)
        gcheb = Chebfun2.from_function(g, domain=dom)
        F1 = Chebfun2v.from_functions(f, g, domain=dom)
        F3 = Chebfun2v(components=[fcheb.approx, gcheb.approx])
        assert _maxdiff(F1, F3, dom) < TOL
