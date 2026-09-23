"""Advection-diffusion in the unit ball.

Translation of sphere/AdvectionDiffusion.m by Nicolas Boulle
(July 2019): the advection-diffusion equation

    c_t = D lap(c) - v . grad(c)

in the unit ball with D = 1/5000 and the divergence-free no-slip
field v = curl[z e^{-5 r^2} (x,y,z)], integrated to t = 15 with
IMEX-BDF1 (a ballfun Helmholtz solve with Neumann conditions each
step).

Original: https://www.chebfun.org/examples/sphere/AdvectionDiffusion.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.ballfun.ballfunv import Ballfunv
from chebfunjax.plotting import PARULA, chebfun_style, plot_ball_slices, quiver_ball
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.quadrature import chebpts

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'sphere')
FIG = [0]


def _ball_slice(c, ax, clim=None):
    """MATLAB @ballfun/slice.m: cross sections at x = 0, y = 0, z = 0."""
    m, n, p = map(int, c.shape)
    m, n, p = max(m, 25), max(n, 28), max(p, 28)
    m += (1 - m % 6) % 6
    n += (4 - n % 4) % 4
    p += (4 - p % 4) % 4
    r = np.asarray(chebpts(m))[m // 2:]
    lam = np.linspace(-np.pi, np.pi, n)
    th = np.linspace(0, np.pi, p)
    rs, rc = np.outer(r, np.sin(th)), np.outer(r, np.cos(th))
    zero = np.zeros_like(rs)

    def ev(lam_, th_):
        return np.real(np.asarray(c.fevalm(jnp.asarray(r), jnp.asarray(lam_),
                                           jnp.asarray(th_))))

    surfaces = [
        (zero, -rs, rc, ev([-np.pi / 2], th)[:, 0, :]),
        (zero, rs, rc, ev([np.pi / 2], th)[:, 0, :]),
        (-rs, zero, rc, ev([np.pi], th)[:, 0, :]),
        (rs, zero, rc, ev([0.0], th)[:, 0, :]),
        (np.outer(r, np.cos(lam)), np.outer(r, np.sin(lam)),
         np.zeros((len(r), len(lam))), ev(lam, [np.pi / 2])[:, :, 0]),
    ]
    if clim is None:
        allv = np.concatenate([s[3].ravel() for s in surfaces])
        clim = (allv.min(), allv.max())
    norm = Normalize(*clim)
    # One Poly3DCollection for all five surfaces so matplotlib depth-sorts
    # the intersecting planes per quad; headlight shading ~ MATLAB
    # camlight('headlight'), lighting phong, material dull.
    el, az = np.deg2rad(30), np.deg2rad(-127.5)
    eye = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                    np.sin(el)])
    quads, colors = [], []
    for xs, ys, zs, cd in surfaces:
        P = np.stack([xs, ys, zs], axis=-1)
        q = np.stack([P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]],
                     axis=2)
        nrm = np.cross(q[:, :, 1] - q[:, :, 0], q[:, :, 3] - q[:, :, 0])
        nn = np.linalg.norm(nrm, axis=-1)
        # Quads collapsed at r = 0 have no normal: leave them unshaded.
        shade = np.where(nn > 1e-14, 0.3 + 0.7 * np.abs(nrm @ eye)
                         / np.maximum(nn, 1e-300), 1.0)
        cq = 0.25 * (cd[:-1, :-1] + cd[1:, :-1] + cd[1:, 1:] + cd[:-1, 1:])
        rgb = PARULA(norm(cq))[..., :3] * shade[..., None]
        quads.append(q.reshape(-1, 4, 3))
        colors.append(rgb.reshape(-1, 3))
    ax.add_collection3d(Poly3DCollection(
        np.concatenate(quads), facecolors=np.concatenate(colors),
        edgecolors=np.concatenate(colors), linewidths=0.3))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=30, azim=-127.5)     # MATLAB view(3)
    ax.set_axis_off()
    return ScalarMappable(norm=norm, cmap=PARULA)


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(
        _IMG, f"AdvectionDiffusion_{FIG[0]:02d}.png"))
    plt.close(fig)


def _slice_plot(c, title, clim=(-0.2, 0.2)):
    """clf, slice(c), caxis(clim), title, colorbar, axis off."""
    fig = plt.figure(figsize=(6.0, 2.7))
    ax = fig.add_axes([0.2, 0.0, 0.6, 0.92], projection="3d")
    sm = _ball_slice(c, ax, clim)
    ax.set_title(title, pad=0)
    cax = fig.add_axes([0.84, 0.11, 0.025, 0.82])
    fig.colorbar(sm, cax=cax)
    _save(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # The divergence-free, no-slip velocity field.
    w = Ballfunv.from_functions(
        lambda x, y, z: z * np.exp(-5 * (x**2 + y**2 + z**2)) * x,
        lambda x, y, z: z * np.exp(-5 * (x**2 + y**2 + z**2)) * y,
        lambda x, y, z: z * np.exp(-5 * (x**2 + y**2 + z**2)) * z)
    v = w.curl()

    # quiver(v, 4, 'numpts', 30), axis('off'), colorbar
    fig = plt.figure(figsize=(6.0, 2.7))
    ax = fig.add_axes([0.2, 0.0, 0.6, 1.0], projection="3d")
    quiver_ball(v, ax=ax, n_pts=30, arrow_scale=4)
    ax.set_axis_off()
    vx_, vy_, vz_ = v.components
    rr = np.linspace(0, 1, 30)
    mags = np.sqrt(sum(np.abs(np.asarray(f_.fevalm(
        jnp.asarray(rr), jnp.asarray(np.linspace(-np.pi, np.pi, 30)),
        jnp.asarray(np.linspace(0, np.pi, 30))))) ** 2
        for f_ in (vx_, vy_, vz_)))
    cax = fig.add_axes([0.84, 0.11, 0.025, 0.82])
    fig.colorbar(ScalarMappable(norm=Normalize(0, float(mags.max())),
                                cmap=PARULA), cax=cax)
    _save(fig)
    print("ans =")
    print(f"     {float(v.div().norm()):.15e}")

    # No-slip: v . n on the boundary r = 1.
    lam = np.linspace(-np.pi, np.pi, 181)
    th = np.linspace(0, np.pi, 91)
    L, T = np.meshgrid(lam, th)
    x = np.cos(L) * np.sin(T)
    y = np.sin(L) * np.sin(T)
    z = np.cos(T)
    r1 = np.ones_like(L)
    vx, vy, vz = v.components
    vn = (np.asarray(vx(r1, L, T)) * x
          + np.asarray(vy(r1, L, T)) * y
          + np.asarray(vz(r1, L, T)) * z)
    print("ans =")
    print(f"     {np.max(np.abs(vn)):.15e}")

    # Initial condition and its visualization.
    c = Ballfun.from_function(
        lambda x_, y_, z_: -x_ * np.exp(-5 * (x_**2 + y_**2 + z_**2)))
    # plot / slice / WedgeAz / WedgePol in a 2x2 subplot grid.
    fig = plt.figure(figsize=(6.0, 2.7))
    for k, (ttl, style) in enumerate([("Plot", "ball"), ("Slice", None),
                                      ("WedgeAz", "WedgeAz"),
                                      ("WedgePol", "WedgePol")]):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        if style is None:
            _ball_slice(c, ax, clim=(-0.19, 0.19))
        else:
            plot_ball_slices(c, ax=ax, style=style, azim=-127.5)
            ax.set_axis_off()
        ax.set_title(ttl, pad=0)
    _save(fig)

    # IMEX-BDF1 to t = 15 (Helmholtz solve with Neumann BC per step).
    D = 1 / 5000
    dt = 0.1
    K = 1j * np.sqrt(1 / (dt * D))
    nsteps = int(np.ceil(15 / dt))
    t0 = time.time()
    for n in range(nsteps + 1):
        if n % 50 == 0:
            _slice_plot(c, f"Time {n * dt:g}")
            print(f"t={n * dt:g} plotted ({time.time()-t0:.0f}s)",
                  flush=True)
        gx, gy, gz = c.grad()
        rhs = K**2 * c + v.dot(Ballfunv(gx, gy, gz)) * (1 / D)
        # Per-step simplification chops the roundoff-seeded parasitic
        # mode of the explicit advection term (growth ~1.5x/step from
        # 1e-16 blows up around step 90 without it) -- MATLAB's ballfun
        # pipeline simplifies adaptively and is stable the same way.
        c = Ballfun.helmholtz(rhs, K, lambda lam_, th_: 0.0, 100,
                              bc_type="neumann").simplify()
    print(f"done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    run()
