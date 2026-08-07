"""Frequencies of a drum.

Faithful replica of ode-eig/Drum.m by Toby Driscoll (November 2010):
the axisymmetric drum eigenvalue problem

    u'' + u'/r = -omega^2 u,   u'(0) = 0, u(1) = 0,

multiplied through by r into the generalized pencil A u = lam B u,
validated against the zeros of J_0; then a variable density
rho = 1 - a sin(pi r) designed by chebfun rootfinding so that
omega2/omega1 = 2, a perfect octave.

Original: https://www.chebfun.org/examples/ode-eig/Drum.html
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
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-eig')


def _pencil():
    A = Chebop(lambda r, u: r * u.diff(2) + u.diff(), domain=(0, 1))
    A.lbc = "neumann"
    A.rbc = "dirichlet"
    return A


def _solve(rho_fun=None, k=6):
    A = _pencil()
    if rho_fun is None:
        B = Chebop(lambda r, u: r * u, domain=(0, 1))
    else:
        B = Chebop(lambda r, u: r * rho_fun(r) * u, domain=(0, 1))
    V, D = A.eigs_generalized(B, k=k)
    lam = np.asarray(D)
    omega = np.sqrt(-lam.real)
    idx = np.argsort(omega)
    # Unit L2 normalization (MATLAB's eigs convention).
    out = []
    for i in idx:
        f = V[i]
        nrm = float(f.norm(2))
        out.append(f * (1.0 / nrm) if nrm > 0 else f)
    return omega[idx], out


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # Constant density: omegas are the zeros of J_0.
    omega, V = _solve()
    print("omega =")
    for w in omega:
        print(f"  {w:.15f}")
    x20 = chebfun(lambda t: t, domain=(0.0, 20.0))
    jroots = np.sort(np.asarray(x20.besselj(0).roots()))
    print("err =")
    for e in omega - jroots[:6]:
        print(f"   {e:.3e}")

    # Drum deflections for pure frequencies.
    rr, tt = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 2 * np.pi, 60))
    fig = plt.figure(figsize=(9.0, 7.0))
    for k in range(4):
        vk = V[k]
        prof = np.asarray(vk(np.linspace(0, 1, 40))).real
        if prof[0] < 0:
            prof = -prof
        Z = np.tile(prof, (60, 1))
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        ax.plot_surface(rr * np.cos(tt), rr * np.sin(tt), Z,
                        cmap="viridis", rstride=1, cstride=1,
                        edgecolor="none", vmin=-3, vmax=3)
        ax.set_zlim(-1, 3)
        ax.view_init(20, -33)
        ax.set_axis_off()
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Drum_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Variable density: design rho = 1 - a sin(pi r) so omega2/omega1 = 2.
    def evratio(a):
        a = float(a)
        omega, _ = _solve(lambda r: 1 - a * (np.pi * r).sin(), k=2)
        return omega[1] / omega[0]

    ratfun = chebfun(
        lambda aa: np.array([evratio(t) for t in np.atleast_1d(aa)],
                            dtype=np.float64),
        domain=(0.5, 1.0), eps=1e-11)
    astar = float(np.asarray((ratfun - 2.0).roots())[0])
    print("astar =")
    print(f"   {astar:.15f}")

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    aa = np.linspace(0.5, 1, 500)
    ax.plot(aa, np.asarray(ratfun(aa)), lw=1.6)
    ax.set_title("Eigenvalue ratio")
    ax.set_xlabel("a")
    ax.set_xticks([0.5, astar, 1.0])
    ax.set_yticks([2.0])
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Drum_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    residual = evratio(astar) - 2
    print("residual =")
    print(f"    {residual:.15e}")

    # Eigenfunctions of the designed drum.
    _, V2 = _solve(lambda r: 1 - astar * (np.pi * r).sin(), k=2)
    fig = plt.figure(figsize=(9.6, 4.6))
    for k in range(2):
        prof = np.asarray(V2[k](np.linspace(0, 1, 40))).real
        Z = np.tile(-prof, (60, 1))
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot_surface(rr * np.cos(tt), rr * np.sin(tt), Z,
                        cmap="copper", linewidth=0)
        ax.set_title(["First mode", "Second mode"][k])
        ax.view_init(20, -33)
        ax.set_axis_off()
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Drum_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
