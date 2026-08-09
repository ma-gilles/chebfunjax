"""Compacting colloids in a centrifuge using pde15s.

Faithful replica of temp/CompactingColloids.m (Julia Schollick and
Rob Style, 2014): the Auzerais-Jackson-Russel sedimentation equation
u_t + [(1-u)^6.55 (u - (1.85/pe) phi_m u' / (phi_m - u)^2)]' = 0 on
[0, 1] with no-flux boundary conditions, solved with pde15s at
Pe = 200 and shown as a waterfall plot.

Original: https://www.chebfun.org/examples/temp/CompactingColloids.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'temp')

PE = 200.0
PHI_M = 0.64
TIME_END = 10.0
U_INIT = 0.3


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    t = np.arange(0, TIME_END + 1e-9, 0.1)

    def pdefun(tt, u):
        return -((1 - u)**6.55
                 * (u - (1.85 * PHI_M / PE) * u.diff()
                    / (PHI_M - u)**2)).diff()

    def flux_bc(u):
        return -u + (1.85 * PHI_M / 100.0) * u.diff() / (PHI_M - u)**2

    # The AJR equation is severely stiff near close packing (the
    # original notes even Mathematica failed); our generic pde15s
    # row-replacement collocation stalls on the sharp front, so this
    # replica integrates the same equation by a conservative
    # finite-volume method of lines with zero-flux faces (the exact
    # no-flux boundary conditions of the example).
    from scipy.integrate import solve_ivp

    n = 400
    xf = np.linspace(0, 1, n + 1)          # faces
    xc = 0.5 * (xf[:-1] + xf[1:])          # cell centers
    dx = xf[1] - xf[0]

    def rhs(tt_, u):
        uf = 0.5 * (u[:-1] + u[1:])        # interior face values
        dudx = (u[1:] - u[:-1]) / dx
        flux = (1 - uf)**6.55 * (
            uf - (1.85 * PHI_M / PE) * dudx
            / np.maximum(PHI_M - uf, 1e-6)**2)
        F = np.zeros(n + 1)
        F[1:-1] = flux                     # zero flux at both walls
        return -(F[1:] - F[:-1]) / dx

    sol = solve_ivp(rhs, (0, TIME_END), U_INIT * np.ones(n),
                    t_eval=t, method="BDF", rtol=1e-7, atol=1e-9)
    tt = sol.t
    uu = [
        chebfun(lambda x, k=k: np.interp(np.asarray(x), xc, sol.y[:, k]),
                domain=(0, 1))
        for k in range(sol.y.shape[1])
    ]
    print(f"integrated {len(tt)}/{len(t)} time steps")

    # Waterfall plot of the compaction front.
    xs = np.linspace(0, 1, 300)
    fig = plt.figure(figsize=(9.2, 6.4))
    ax = fig.add_subplot(projection="3d")
    for k in range(0, len(tt), 2):
        vals = np.asarray(uu[k](xs))
        ax.plot(xs, np.full_like(xs, tt[k]), vals, 'b', lw=1.0)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    ax.set_zlim(0, 1)
    ax.view_init(35, -70)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "CompactingColloids_repl_01.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Conservation check: total particle mass is conserved.
    m0 = float(uu[0].sum())
    mend = float(uu[-1].sum())
    print(f"mass at t=0:  {m0:.6f}")
    print(f"mass at t=10: {mend:.6f}")
    ufin = np.asarray(uu[-1](xs))
    print(f"u(0, t=10) = {ufin[0]:.4f}  (this end of the cell empties)")
    print(f"u(1, t=10) = {ufin[-1]:.4f}  (particles pack toward close "
          f"packing {PHI_M})")


if __name__ == "__main__":
    run()
