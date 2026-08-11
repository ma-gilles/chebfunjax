"""Port of MATLAB Chebfun tests/chebfun2/test_vertcat.m (Fable 5).

FIXED (Fable 5, chebfun2/3 skip sweep): ``Chebfun2.vertcat()`` now
stacks Chebfun2s into a Chebfun2v.  MATLAB's ``[f; g]`` bracket syntax
has no Python counterpart, so the explicit call is used.

Provenance
----------
MATLAB source : tests/chebfun2/test_vertcat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

EPS = float(np.finfo(np.float64).eps)
TOL = 10 * EPS


class TestChebfun2Vertcat:
    def test_vertcat_unit_square(self):
        # pass(1): [x; y] equals chebfun2v(x, y).
        f = Chebfun2.from_function(lambda x, y: x)
        g = Chebfun2.from_function(lambda x, y: y)
        F = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y)
        assert float((f.vertcat(g) - F).norm()) < TOL

    def test_vertcat_on_rectangle(self):
        # pass(2, 3): the same on [-1 0 -2 1].
        d = (-1.0, 0.0, -2.0, 1.0)
        f = Chebfun2.from_function(lambda x, y: x, domain=d)
        g = Chebfun2.from_function(lambda x, y: y, domain=d)
        F = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y,
                                     domain=d)
        assert float((f.vertcat(g) - F).norm()) < TOL

    def test_vertcat_single_argument(self):
        # pass(4): vertcat of one component is that component.
        d = (-1.0, 0.0, -2.0, 1.0)
        f = Chebfun2.from_function(lambda x, y: x, domain=d)
        assert float((f.vertcat() - f).norm()) < TOL

    def test_vertcat_three_components(self):
        # A 3-component stack is the Chebfun2v used for 3D vector fields.
        f = Chebfun2.from_function(lambda x, y: x)
        g = Chebfun2.from_function(lambda x, y: y)
        h = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        assert f.vertcat(g, h).n_components == 3
