"""Port of MATLAB Chebfun tests/spinopsphere/test_spinopsphere.m (Fable 5).

Provenance
----------
MATLAB source : tests/spinopsphere/test_spinopsphere.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.operators.spinopsphere import Spinopsphere, func2str


class TestSpinopsphereSpinopsphere:
    def test_all_matlab_assertions(self):
        # Construction from STRING for the GL equation:
        #   S = spinopsphere('GL');  N = S.nonlin;
        #   pass(1) = strcmpi(func2str(N), '@(u)u-(1+1.5i)*u.*(abs(u).^2)');
        S = Spinopsphere("GL")
        N = S.nonlin
        assert func2str(N) == "@(u)u-(1+1.5i)*u.*(abs(u).^2)"

        # Construction from TSPAN:
        #   tspan = [0 1];  S = spinopsphere(tspan);
        #   pass(2) = isequal(S.tspan, tspan);
        tspan = [0, 1]
        S = Spinopsphere(tspan)
        assert S.tspan == (0.0, 1.0)
