"""Port of MATLAB Chebfun tests/spinop2/test_spin2.m (Fable 5).

The MATLAB test is a self-convergence check: solve the GL equation with
steps ``dt`` and ``dt/2`` at ``N=128``, then compare on a 50x50 grid. Its
source tolerance is ``1e-4``. MATLAB saves and restores the random state
between solves, so both use the same initial condition. This port likewise
constructs one GL preset and reuses it for both solves. The preset creates
its initial field with the JAX ``randnfun2`` implementation; this test does
not claim MATLAB seeded-normal-stream parity.

Provenance
----------
MATLAB source : tests/spinop2/test_spin2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.operators.spinop2 import Spinop2, spin2

TOL = 1e-4  # MATLAB test tolerance (do not widen)


class TestSpinop2Spin2:
    def test_all_matlab_assertions(self):
        # ---- GL ------------------------------------------------------
        #   S = spinop2('GL'); S.tspan = S.tspan/10;
        #   N = 128; dt = 1e-1;
        #   s = rng; u = spin2(S, N, dt, 'plot', 'off');
        #   rng(s); v = spin2(S, N, dt/2, 'plot', 'off');
        S = Spinop2("GL")
        S.tspan = tuple(t / 10 for t in S.tspan)
        N = 128
        dt = 1e-1
        u = spin2(S, N, dt, "plot", "off")
        v = spin2(S, N, dt / 2, "plot", "off")

        # Compare on the MATLAB 50x50 grid over [dom(1), dom(2)]^2:
        #   [xx, yy] = meshgrid(linspace(dom(1), dom(2), 50));
        #   scale = max(max(abs(v(xx,yy))));
        #   pass(1) = max(max(abs(u(xx,yy)-v(xx,yy))))/scale < tol;
        dom = S.domain
        lam = np.linspace(dom[0], dom[1], 50)
        xx, yy = np.meshgrid(lam, lam)
        vv = v(xx, yy)
        scale = float(np.max(np.abs(vv)))
        err = float(np.max(np.abs(u(xx, yy) - vv))) / scale
        assert err < TOL, f"GL self-convergence {err} !< {TOL}"
