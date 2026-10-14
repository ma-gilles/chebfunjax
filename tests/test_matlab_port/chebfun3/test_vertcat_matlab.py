"""Port of MATLAB Chebfun tests/chebfun3/test_vertcat.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_vertcat.m
Chebfun commit: 7574c77

The four original predicates call the public scalar vertcat method.
"""

from __future__ import annotations

import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v
from chebfunjax.chebpref import ChebfunPref

TOL = 10 * ChebfunPref().cheb3Prefs.chebfun3eps

DOMS = [(-1, 1, -1, 1, -1, 1), (-1, 0, -2, 1, 2, 4)]


class TestChebfun3Vertcat:
    @pytest.mark.parametrize("dom", DOMS)
    def test_vertcat_three(self, dom):
        f = Chebfun3.from_function(lambda x, y, z: x, domain=dom)
        g = Chebfun3.from_function(lambda x, y, z: y, domain=dom)
        h = Chebfun3.from_function(lambda x, y, z: z, domain=dom)
        F = Chebfun3v.from_functions(lambda x, y, z: x, lambda x, y, z: y,
                                     lambda x, y, z: z, domain=dom)
        actual = f.vertcat(g, h)
        assert float((actual - F).norm()) < TOL  # native pass(1)/(2)
        if dom == DOMS[1]:
            assert float((actual - Chebfun3v([f, g, h])).norm()) < TOL  # pass(3)
            assert float((f.vertcat() - f).norm()) < TOL  # native pass(4)

    def test_source_dispatch_and_identity(self):
        f = Chebfun3.from_function(lambda x, y, z: x)
        g = Chebfun3.from_function(lambda x, y, z: y)
        h = Chebfun3.from_function(lambda x, y, z: z)
        assert f.vertcat() is f
        assert Chebfun3.empty().vertcat().isempty()
        pair = f.vertcat(g)
        assert pair.components[0] is f and pair.components[1] is g
        triple = h.vertcat(pair)
        assert all(a is b for a, b in zip(triple.components, (h, f, g)))
        # Literal source second-vector branch ignores a third input.
        assert all(a is b for a, b in zip(h.vertcat(pair, f).components,
                                         (h, f, g)))
        with pytest.raises(ValueError, match="vertcat:tooManyComponents"):
            f.vertcat(Chebfun3v(f, g, h))
        with pytest.raises(ValueError, match="vertcat:tooManyInputs"):
            f.vertcat(f, g, h)
        with pytest.raises(ValueError, match="vertcat:badInputs"):
            f.vertcat(2)
        with pytest.raises(IndexError):
            f.vertcat(Chebfun3v(g))
