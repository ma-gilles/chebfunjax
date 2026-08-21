"""Port of MATLAB Chebfun tests/chebmatrix/test_changeTech.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebmatrix/test_changeTech.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.operators.chebmatrix import ChebMatrix

jax.config.update("jax_enable_x64", True)

TOL = 1e-13
DOM = (0.0, 2 * np.pi)


def _tech_name(f):
    return type(f.funs[0].tech).__name__.lower()


def _maxdiff(f, g):
    xs = jnp.linspace(DOM[0] + 1e-9, DOM[1] - 1e-9, 33)
    return float(jnp.max(jnp.abs(jnp.asarray(f(xs))
                                 - jnp.asarray(g(xs)))))


class TestChebmatrixChangetech:
    def test_cheb_to_trig(self):
        f1 = cj.chebfun(lambda x: jnp.cos(x), domain=DOM)
        f2 = cj.chebfun(lambda x: jnp.sin(x), domain=DOM)
        F = ChebMatrix.vertcat(f1, f2)
        G = F.change_tech("trigtech")
        assert "trig" in _tech_name(G[0, 0])  # pass(1)
        assert "trig" in _tech_name(G[1, 0])  # pass(2)
        assert _maxdiff(f1, G[0, 0]) < TOL  # pass(3)
        assert _maxdiff(f2, G[1, 0]) < TOL  # pass(4)

    def test_trig_to_cheb(self):
        f1 = cj.chebfun(lambda x: jnp.cos(x), domain=DOM, trig=True)
        f2 = cj.chebfun(lambda x: jnp.sin(x), domain=DOM, trig=True)
        F = ChebMatrix.vertcat(f1, f2)
        G = F.change_tech("chebtech2")
        assert "trig" not in _tech_name(G[0, 0])  # pass(5)
        assert "trig" not in _tech_name(G[1, 0])  # pass(6)
        assert _maxdiff(f1, G[0, 0]) < TOL  # pass(7)
        assert _maxdiff(f2, G[1, 0]) < TOL  # pass(8)

    def test_mixed_objects(self):
        f2 = cj.chebfun(lambda x: jnp.cos(10 * x), domain=DOM,
                        trig=True)
        f3 = cj.chebfun(lambda x: jnp.sin(10 * x), domain=DOM)
        F = ChebMatrix.vertcat(1.0, f2, f3)
        G = F.change_tech("trigtech")
        # A scalar must stay a scalar.  pass(9)/(10)
        assert isinstance(G[0, 0], (int, float))
        assert G[0, 0] == 1.0
        # Already the right tech: returned unchanged.  pass(11)
        assert G[1, 0] is f2
        # Converted.  pass(12)/(13)
        assert "trig" in _tech_name(G[2, 0])
        assert _maxdiff(f3, G[2, 0]) < TOL
