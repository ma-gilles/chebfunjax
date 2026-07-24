"""Port of MATLAB Chebfun tests/spinop2/test_spin2.m (Fable 5).

The MATLAB test is a self-convergence (Cauchy) check: solve the GL
equation with time-step DT and DT/2 at N=128 grid points, then compare
the two solutions on a 50x50 grid.  MATLAB tolerance ``tol = 1e-4``.

MATLAB seeds a shared random initial condition via ``rng``/``randnfun2``;
chebfunjax has no ``randnfun2``, so the GL preset supplies a
deterministic low-frequency trig field instead (both solves share it,
which is exactly what the self-convergence check needs).

Measured self-convergence error (this port, fully deterministic):
    GL: 7.9e-5  (the genuine 4th-order ETDRK4 error at N=128, dt=0.1 --
        essentially independent of the init amplitude, and the same
        error MATLAB incurs, hence the shared 1e-4 threshold is kept
        as-is; the result is bit-reproducible so there is no flakiness
        despite the sub-2x nominal margin).

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
