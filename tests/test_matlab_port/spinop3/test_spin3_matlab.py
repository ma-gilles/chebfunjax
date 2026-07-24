"""Port of MATLAB Chebfun tests/spinop3/test_spin3.m (Fable 5).

The MATLAB test is a self-convergence (Cauchy) check: solve the GL
equation with time-step DT and DT/2 at N=32 grid points, then compare
the two solutions on a 20x20x20 trig tensor grid.  MATLAB tolerance
``tol = 1e-2``.

MATLAB seeds a shared random initial condition via ``rng``/random
``chebfun3(vals, dom, 'trig')``; chebfunjax has no random 3D trig
constructor, so the GL preset supplies a deterministic low-frequency
trig field (both solves share it, which is what the self-convergence
check needs).

Measured self-convergence error (this port, fully deterministic):
    GL: 7.4e-3  (the genuine 4th-order ETDRK4 error at N=32, dt=0.1 --
        the same error MATLAB incurs, hence the shared 1e-2 threshold
        is kept as-is; bit-reproducible, so no flakiness despite the
        sub-2x nominal margin).

Provenance
----------
MATLAB source : tests/spinop3/test_spin3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.operators.spinop3 import Spinop3, spin3

TOL = 1e-2  # MATLAB test tolerance (do not widen)


class TestSpinop3Spin3:
    def test_all_matlab_assertions(self):
        # ---- GL ------------------------------------------------------
        #   S = spinop3('GL'); S.tspan = S.tspan/10;
        #   N = 32; dt = 1e-1;
        #   s = rng; u = spin3(S, N, dt, 'plot', 'off');
        #   rng(s); v = spin3(S, N, dt/2, 'plot', 'off');
        S = Spinop3("GL")
        S.tspan = tuple(t / 10 for t in S.tspan)
        N = 32
        dt = 1e-1
        u = spin3(S, N, dt, "plot", "off")
        v = spin3(S, N, dt / 2, "plot", "off")

        # Compare on the MATLAB 20x20x20 trig tensor grid over dom:
        #   pts = trigtech.tensorGrid([20 20 20], dom);
        #   xx = pts{1}; yy = pts{2}; zz = pts{3};
        #   scale = max(max(max(abs(v(xx,yy,zz)))));
        #   pass(1) = max(...abs(u(xx,yy,zz)-v(xx,yy,zz))...)/scale < tol;
        # trigtech points are equispaced, endpoint-excluded, over each
        # interval of the (cubic) domain.
        dom = S.domain
        p = np.linspace(dom[0], dom[1], 20, endpoint=False)
        xx, yy, zz = np.meshgrid(p, p, p, indexing="ij")
        vv = v(xx, yy, zz)
        scale = float(np.max(np.abs(vv)))
        err = float(np.max(np.abs(u(xx, yy, zz) - vv))) / scale
        assert err < TOL, f"GL self-convergence {err} !< {TOL}"
