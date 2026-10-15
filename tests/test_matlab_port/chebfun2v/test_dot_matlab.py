"""Port of MATLAB Chebfun tests/chebfun2v/test_dot.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_dot.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebpref import ChebfunPref

X0, Y0 = jnp.asarray(0.3), jnp.asarray(-0.4)


class TestChebfun2vDot:
    def test_position_dot(self):
        P = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y)
        d = P.dot(P)
        exact = float(X0) ** 2 + float(Y0) ** 2
        assert abs(float(d(X0, Y0)) - exact) < 1e-10


@pytest.mark.parametrize("count", [2, 3])
def test_original_definition(count):
    fns = [lambda x, y: jnp.cos(x), lambda x, y: jnp.sin(y), lambda x, y: x*y]
    gns = [lambda x, y: x, lambda x, y: y, lambda x, y: x+y]
    field = Chebfun2v.from_functions(*fns[:count])
    other = Chebfun2v.from_functions(*gns[:count])
    # MATLAB F' * G maps to conjugate transpose followed by Python @.
    expected = field.ctranspose() @ other
    actual = Chebfun2(approx=field.dot(other))
    assert float((actual-expected).norm()) < 50*ChebfunPref().cheb2Prefs.chebfun2eps
