"""Port of MATLAB Chebfun tests/chebfun2/test_cumsum.m (Fable 5).

FIXED: Chebfun2.cumsum/cumsum2 added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun2/test_cumsum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

TOL = 100 * np.finfo(float).eps


def _maxdiff(f, g, dom):
    xs = jnp.asarray(np.linspace(dom[0], dom[1], 13))
    ys = jnp.asarray(np.linspace(dom[2], dom[3], 13))
    xx, yy = jnp.meshgrid(xs, ys, indexing="ij")
    return float(jnp.max(jnp.abs(f(xx, yy) - g(xx, yy))))


class TestChebfun2Cumsum:
    def test_square_domain(self):
        dom = (-1.0, 1.0, -1.0, 1.0)
        x = Chebfun2.from_function(lambda x, y: x, domain=dom)
        y = Chebfun2.from_function(lambda x, y: y, domain=dom)
        assert _maxdiff(x.cumsum(),
                        lambda X, Y: X * (Y + 1), dom) < TOL
        assert _maxdiff(x.cumsum(2),
                        lambda X, Y: 0.5 * (X ** 2 - 1), dom) < TOL
        assert _maxdiff(y.cumsum(2),
                        lambda X, Y: Y * (X + 1), dom) < TOL
        assert _maxdiff(y.cumsum(),
                        lambda X, Y: 0.5 * (Y ** 2 - 1), dom) < TOL

    def test_rectangular_domain(self):
        dom = (-1.1, 2.0, -0.2, 3.0)
        x = Chebfun2.from_function(lambda x, y: x, domain=dom)
        y = Chebfun2.from_function(lambda x, y: y, domain=dom)
        assert _maxdiff(x.cumsum(),
                        lambda X, Y: X * (Y + 0.2), dom) < 10 * TOL
        assert _maxdiff(x.cumsum(2),
                        lambda X, Y: 0.5 * (X ** 2 - 1.1 ** 2),
                        dom) < 10 * TOL
        assert _maxdiff(y.cumsum(2),
                        lambda X, Y: Y * (X + 1.1), dom) < 10 * TOL
        assert _maxdiff(y.cumsum(),
                        lambda X, Y: 0.5 * (Y ** 2 - 0.2 ** 2),
                        dom) < 10 * TOL

    def test_double_cumsum_is_cumsum2(self):
        dom = (-1.1, 2.0, -0.2, 3.0)
        f = Chebfun2.from_function(
            lambda X, Y: jnp.sin((X - 0.1) * (Y + 0.1)), domain=dom)
        assert _maxdiff(f.cumsum().cumsum(2), f.cumsum2(),
                        dom) < 10 * TOL
        assert _maxdiff(f.cumsum(2).cumsum(), f.cumsum2(),
                        dom) < 10 * TOL
