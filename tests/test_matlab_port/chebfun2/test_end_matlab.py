"""Port of MATLAB Chebfun tests/chebfun2/test_end.m (Fable 5).

MATLAB's ``end`` in a Chebfun2 subscript is the string ``"end"``; ``:``
is ``":"``.

Provenance
----------
MATLAB source : tests/chebfun2/test_end.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun2d.chebfun2 import chebfun2

jax.config.update("jax_enable_x64", True)


class TestChebfun2End:
    def test_all_matlab_assertions(self):
        u = chebfun2(lambda x, y: x + y, domain=(0.0, 2.0, 0.0, 3.0))
        assert abs(float(u("end", "end")) - float(u(2.0, 3.0))) < 1e-14  # pass(1)
        x = chebfun(lambda t: t, domain=(0.0, 2.0))
        f = x.T + 3
        assert float((u(":", "end") - f).norm(2)) < 1e-14              # pass(2)
        y = chebfun(lambda t: t, domain=(0.0, 3.0))
        g = 2 + y
        assert float((u("end", ":") - g).norm(2)) < 1e-14              # pass(3)
