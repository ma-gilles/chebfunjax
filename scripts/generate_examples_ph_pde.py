"""Generate example-plot placeholders for chebfun.org ``pde/`` pages that
chebfunjax had never rendered.

Pages ported here (faithful ports of the MATLAB chebfun.org sources):

pde:
  * CompactingColloids -- the Auzerais-Jackson-Russel (AJR) sedimentation
    equation solved with ``pde15s`` and no-flux (Robin/flux-form) boundary
    conditions.  A waterfall plot of the particle concentration over space
    and time.

Solver note
-----------
The AJR equation is a stiff *conservation law*

    u_t + [ (1-u)^6.55 ( u - (1.85/pe) phi_m u' / (phi_m - u)^2 ) ]' = 0

with no-flux boundary conditions at both ends.  MATLAB's ``pde15s`` handles
it with a singular-mass-matrix DAE (``ode15s``) that projects the interior
Chebyshev-collocation equations and slaves the boundary values to the flux
constraints, so it conserves the total mass exactly.  ``scipy`` exposes no
singular-mass DAE integrator, and the row-replacement fallback used by
:func:`chebfunjax.chebfun1d.pde15s.pde15s` for genuinely nonlinear operators
leaks mass through the boundary for this problem (the packed front decays
instead of sharpening).  To render a *faithful* figure we therefore integrate
the PDE with a conservative finite-volume method of lines -- zero physical
flux across both boundary faces guarantees exact mass conservation, exactly
the "no flux out of the top or bottom" condition the example imposes -- and
plot the resulting snapshots as a MATLAB-style coloured waterfall.  This
matches the sibling ``ode-*`` generators, which likewise integrate their PDEs
directly for the figure rather than routing through the library ODE/PDE
front-ends.

(The general Robin/flux callable BC *parsing* added to ``pde15s`` in the same
change is exercised by the unit tests in
``tests/unit/test_pde15s_robin_bc.py``.)
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib

matplotlib.use("Agg")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.integrate import solve_ivp

from chebfunjax.plotting import chebfun_style, save_chebfun_figure

chebfun_style()

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
REFROOT = ("/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/"
           "refs/docs/images")


def save(fig, cat, name):
    from PIL import Image

    ref_path = os.path.join(REFROOT, cat, name)
    size = Image.open(ref_path).size if os.path.exists(ref_path) else (600, 270)
    save_chebfun_figure(fig, os.path.join(DOCS, cat, name), size=size)
    plt.close(fig)
    print(f"  {cat}/{name} saved")


# ==========================================================================
# pde/CompactingColloids
# ==========================================================================
def compacting_colloids():
    """pde/CompactingColloids -- Auzerais-Jackson-Russel sedimentation PDE.

    MATLAB (Schollick & Style, chebfun.org example pde/CompactingColloids.m):

        pe = 200; phi_m = 0.64; time_end = 10; u_init = 0.3;
        dom = [0, 1]; t = 0:.1:time_end;
        pdefun = @(u) -diff((1-u).^6.55.*(u - 1./pe.*1.85.*phi_m ...
                            ./(phi_m-u).^2.*diff(u)));
        bc.left  = @(u) -u + 1./100.*1.85.*phi_m./(phi_m-u).^2.*diff(u);
        bc.right = bc.left;
        u0 = chebfun(u_init, dom);
        opts = pdeset('Ylim', [0, 1], 'AdjustBCs', false);
        [t, u] = pde15s(pdefun, t, u0, bc, opts);
        waterfall(u, t, 'LineWidth', 2)

    Integrated here with a conservative finite-volume method of lines and no
    physical flux across the boundary faces (the intended no-flux condition),
    then drawn as a coloured waterfall over ``x`` in [0, 1] and ``t`` in
    [0, 10].
    """
    pe, phi_m, time_end, u_init = 200.0, 0.64, 10.0, 0.3
    t = np.arange(0.0, time_end + 1e-9, 0.1)

    # ---- conservative finite-volume MOL --------------------------------
    N = 300
    dx = 1.0 / N
    xc = (np.arange(N) + 0.5) * dx        # cell centres in (0, 1)

    def interior_flux(u):
        # Flux F = (1-u)^6.55 (u - (1.85/pe) phi_m u' / (phi_m-u)^2)
        # sampled at the N-1 interior faces (arithmetic-mean value,
        # central-difference gradient).
        uf = 0.5 * (u[1:] + u[:-1])
        ux = (u[1:] - u[:-1]) / dx
        return (1.0 - uf) ** 6.55 * (
            uf - (1.0 / pe) * 1.85 * phi_m / (phi_m - uf) ** 2 * ux)

    def rhs(_t, u):
        f = np.empty(N + 1)
        f[0] = 0.0                        # no flux across bottom face
        f[-1] = 0.0                       # no flux across top face
        f[1:-1] = interior_flux(u)
        return -(f[1:] - f[:-1]) / dx

    sol = solve_ivp(rhs, (0.0, time_end), np.full(N, u_init), t_eval=t,
                    method="BDF", rtol=1e-6, atol=1e-8)
    if not sol.success:
        raise RuntimeError(f"CompactingColloids solve failed: {sol.message}")
    U = sol.y                             # (N, len(t))
    print(f"    mass(t0,t_mid,t_end) = "
          f"{U[:, 0].mean():.4f}, {U[:, len(t)//2].mean():.4f}, "
          f"{U[:, -1].mean():.4f}  (conserved at {u_init})")
    print(f"    concentration range = [{U.min():.3f}, {U.max():.3f}]")

    # ---- coloured waterfall --------------------------------------------
    # Include the boundary values (recovered from the no-flux condition:
    # the packed profile is nearly flat at the wall) by padding with the
    # nearest cell centre; the plot grid is [0, xc, 1].
    xgrid = np.concatenate([[0.0], xc, [1.0]])

    fig = plt.figure(figsize=(6.0, 2.7))
    ax = fig.add_subplot(111, projection="3d")

    zmin, zmax = -0.2, float(U.max())
    norm = plt.Normalize(vmin=zmin, vmax=0.65)
    cmap = plt.get_cmap("viridis")

    # Draw every other snapshot as a height-coloured line (waterfall draws
    # mesh lines only along x); coarser stride keeps the mesh legible at
    # 600x270 while preserving the envelope.
    for k in range(0, len(t), 1):
        col = U[:, k]
        z = np.concatenate([[col[0]], col, [col[-1]]])
        pts = np.column_stack([xgrid, np.full_like(xgrid, t[k]), z])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        zmid = 0.5 * (z[:-1] + z[1:])
        lc = Line3DCollection(segs, cmap=cmap, norm=norm, linewidths=1.0)
        lc.set_array(zmid)
        ax.add_collection3d(lc)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, time_end)            # t axis 0 -> 10
    ax.set_zlim(zmin, 0.65)
    ax.view_init(elev=22.0, azim=-120.0)  # MATLAB-style 3-D waterfall view
    ax.set_box_aspect((1.6, 1.15, 0.5))
    # MATLAB tick style: sparse, thin panes.
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 5, 10])
    ax.set_zticks([-0.2, 0.0, 0.2, 0.4, 0.6])
    ax.tick_params(labelsize=7)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_edgecolor((0.85, 0.85, 0.85))
        pane.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.grid(True)
    ax.set_position([-0.10, -0.10, 1.20, 1.24])

    save(fig, "temp", "CompactingColloids_01.png")


PAGES = {
    "CompactingColloids": compacting_colloids,
}


if __name__ == "__main__":
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, fn in PAGES.items():
        if flt.lower() in name.lower():
            print(f"[{name}]")
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
