"""Port of MATLAB Chebfun tests/spinop2/test_spinop2.m (Fable 5).

Provenance
----------
MATLAB source : tests/spinop2/test_spinop2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.operators.spinop2 import Spinop2, func2str


class TestSpinop2Spinop2:
    def test_all_matlab_assertions(self):
        # Construction from STRING for GL equation:
        #   S = spinop2('GL');  N = S.nonlin;
        #   pass(1) = strcmpi(func2str(N), '@(u)u-(1+1.5i)*u.*(abs(u).^2)');
        S = Spinop2("GL")
        N = S.nonlin
        assert func2str(N) == "@(u)u-(1+1.5i)*u.*(abs(u).^2)"

        # Construction from DOM/TSPAN:
        #   dom = [0 2*pi 0 2*pi];  tspan = [0 1];
        #   S = spinop2(dom, tspan);
        #   pass(2) = isequal(S.domain, dom);
        #   pass(3) = isequal(S.tspan, tspan);
        dom = [0.0, 2 * np.pi, 0.0, 2 * np.pi]
        tspan = [0.0, 1.0]
        S = Spinop2(dom, tspan)
        assert S.domain == tuple(dom)
        assert S.tspan == (0.0, 1.0)

        # Test recursive SUBSREF:
        #   pass(4) = isequal(S.domain([2 4]), [2*pi 2*pi]);
        # (MATLAB 1-indexed [2 4] -> Python 0-indexed [1, 3].)
        assert (S.domain[1], S.domain[3]) == (2 * np.pi, 2 * np.pi)
