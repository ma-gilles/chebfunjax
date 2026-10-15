"""Port of MATLAB Chebfun tests/chebfun2v/test_vertcat.m (Fable 5).

MATLAB [f; f] / [F; f] concatenation notation maps to constructing a
Chebfun2v from component approx objects.

Provenance
----------
MATLAB source : tests/chebfun2v/test_vertcat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

TOL = 1e3 * float(np.finfo(np.float64).eps)
PTS = [(-0.5, 0.3), (0.2, -0.7), (0.9, 0.9)]


def _err(approx, f):
    fa = Chebfun2(approx=approx)
    return max(abs(float(np.asarray(fa(x, y))) - float(np.asarray(f(x, y))))
               for x, y in PTS)


class TestChebfun2vVertcat:
    def test_concatenations(self):
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        F = Chebfun2v(components=[f.approx, f.approx])
        G = Chebfun2v(components=[f.approx, f.approx, f.approx])
        K = Chebfun2v(components=[f.approx] + list(F.components))
        for V in (F, G, K):
            for comp in V.components:
                assert _err(comp, f) < TOL
        assert len(F.components) == 2
        assert len(G.components) == 3
        assert len(K.components) == 3


@pytest.mark.parametrize("domain", [(-1., 1., -1., 1.), (-3., 2., -1., 2.)])
def test_all_original_function_norm_predicates(domain):
    # Seven assertions on each source domain; use public concatenation and
    # continuous function norms, not selected point-value substitutions.
    f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y), domain=domain)
    F = f.vertcat(f)
    G = f.vertcat(f).vertcat(f)
    H = F.vertcat(f)
    K = f.vertcat(F)
    for component in F.components:
        assert (Chebfun2(approx=component) - f).norm() < TOL
    for component in G.components:
        assert (Chebfun2(approx=component) - f).norm() < TOL
    assert (G - H).norm() < TOL
    assert (G - K).norm() < TOL


def test_prepend_to_three_components_source_error():
    f = Chebfun2.from_function(lambda x, y: x + y)
    three = f.vertcat(f).vertcat(f)
    with pytest.raises(ValueError, match="SEPARABLEAPPROX:vertcat:tooManyComponents"):
        f.vertcat(three)
