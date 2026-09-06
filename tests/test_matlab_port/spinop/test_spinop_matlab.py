"""Port of MATLAB Chebfun tests/spinop/test_spinop.m (Fable 5).

MATLAB ``func2str(S.lin)`` is ``S.lin_str`` (the preset's MATLAB
operator string); ``spinop(dom, tspan)`` is ``Spinop(domain=, tspan=)``.

Provenance
----------
MATLAB source : tests/spinop/test_spinop.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.operators.spinop import Spinop


class TestSpinopSpinop:
    def test_all_matlab_assertions(self):
        S = Spinop("ks")
        assert S.lin_str.lower() == "@(u)-diff(u,2)-diff(u,4)"               # pass(1)
        dom = (0.0, 2 * np.pi)
        tspan = (0.0, 1.0)
        S = Spinop(domain=dom, tspan=tspan)
        assert tuple(S.domain) == dom                                        # pass(2)
        assert tuple(S.tspan) == tspan                                       # pass(3)
        assert S.domain[1] == 2 * np.pi                                      # pass(4)
