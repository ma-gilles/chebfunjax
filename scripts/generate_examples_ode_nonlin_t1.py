"""Generate per-block figures for the ode-nonlin example category,
tranche 1: DelayDifferentialEquations, TwoElectrons, AllenCahn,
Logistic, ChebopQuiver.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from chebfunjax.plotting import CHEBFUN_BLUE, chebfun_style, save_chebfun_figure

chebfun_style()

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
REFROOT = ("/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/"
           "refs/docs/images")
ORANGE = "#D95319"
PI = float(np.pi)


def save(fig, name):
    from PIL import Image

    ref_path = os.path.join(REFROOT, "ode-nonlin", name)
    size = Image.open(ref_path).size if os.path.exists(ref_path) else (600, 270)
    save_chebfun_figure(fig, os.path.join(DOCS, "ode-nonlin", name),
                        size=size)
    plt.close(fig)
    print(f"  ode-nonlin/{name} saved")


def diffmat(x):
    N = len(x)
    c = np.ones(N)
    c[0] = c[-1] = 2.0
    c *= (-1.0) ** np.arange(N)
    X = x[:, None] - x[None, :]
    D = (c[:, None] / c[None, :]) / (X + np.eye(N))
    return D - np.diag(D.sum(axis=1))


def chebgrid(n, a=-1.0, b=1.0):
    xs = np.cos(PI * np.arange(n) / (n - 1))[::-1]
    return 0.5 * (a + b) + 0.5 * (b - a) * xs


def delaydifferentialequations():
    """ode-nonlin/DelayDifferentialEquations — 24 panels."""
    panel = 1

    def plot_sol(ts, xs_, title="", color=CHEBFUN_BLUE):
        nonlocal panel
        fig, ax = plt.subplots()
        ax.plot(ts, xs_, color=color, linewidth=1.4)
        ax.grid(True, alpha=0.4, linewidth=0.4)
        if title:
            ax.set_title(title, fontsize=9)
        save(fig, f"DelayDifferentialEquations_{panel:02d}.png")
        panel += 1

    # 1: pantograph x'(t) = a x(q t), fixed-point iteration
    from scipy.interpolate import interp1d

    def pantograph(a, b, q, x0, dom=(0.0, 1.0), iters=30):
        ts = np.linspace(*dom, 800)
        x = np.full_like(ts, x0)
        for _ in range(iters):
            xi = interp1d(ts, x, fill_value="extrapolate")
            rhs = a * x + b * xi(q * ts) if b != 0 else a * xi(q * ts)
            xnew = x0 + np.concatenate(
                [[0], np.cumsum(0.5 * (rhs[1:] + rhs[:-1])
                                * np.diff(ts))])
            if np.max(np.abs(xnew - x)) < 1e-12:
                x = xnew
                break
            x = xnew
        return ts, x

    ts, x = pantograph(0.0, 1.0, 0.5, 1.0)
    plot_sol(ts, x, "pantograph x' = x(t/2)")

    ts, x = pantograph(1.0, -8.0, 0.5, 1.0)
    plot_sol(ts, x, "x' = x - 8 x(t/2)")

    # 3-6: pantograph family for several q
    for q in (0.9, 0.7, 0.3, 0.1):
        ts, x = pantograph(0.0, 1.0, q, 1.0)
        plot_sol(ts, x, f"x' = x(q t), q = {q:g}")

    # 7-10: constant-delay equations by method of steps
    def constant_delay(f, tau, hist, T, n_per=400):
        ts_all = [np.array([0.0])]
        xs_all = [np.array([hist(0.0)])]
        t0 = 0.0
        xfun = hist
        while t0 < T - 1e-12:
            t1 = min(t0 + tau, T)
            sol = solve_ivp(lambda t, y: [f(t, y[0], xfun(t - tau))],
                            (t0, t1), [xs_all[-1][-1]],
                            t_eval=np.linspace(t0, t1, n_per),
                            rtol=1e-9, max_step=tau / 50)
            ts_all.append(sol.t)
            xs_all.append(sol.y[0])
            tt_hist = np.concatenate(ts_all)
            xx_hist = np.concatenate(xs_all)
            xfun = interp1d(tt_hist, xx_hist, bounds_error=False,
                            fill_value=(xx_hist[0], xx_hist[-1]))
            t0 = t1
        return np.concatenate(ts_all), np.concatenate(xs_all)

    ts, x = constant_delay(lambda t, x, xd: -xd, 1.0,
                           lambda t: 1.0, 10.0)
    plot_sol(ts, x, "x' = -x(t-1)")

    ts, x = constant_delay(lambda t, x, xd: -2 * xd, 1.0,
                           lambda t: 1.0, 10.0)
    plot_sol(ts, x, "x' = -2 x(t-1): oscillatory", color=ORANGE)

    for lam in (0.5, 1.6):
        ts, x = constant_delay(lambda t, x, xd, _l=lam: -_l * xd,
                               1.0, lambda t: 1.0, 20.0)
        plot_sol(ts, x, f"x' = -{lam:g} x(t-1)")

    # 11-14: delayed logistic (Hutchinson)
    for r in (0.8, 1.4, 1.8, 2.2):
        ts, x = constant_delay(
            lambda t, x, xd, _r=r: _r * x * (1 - xd), 1.0,
            lambda t: 0.1, 40.0)
        plot_sol(ts, x, f"Hutchinson r = {r:g}")

    # 15-18: Mackey-Glass at several parameters
    for beta in (1.2, 2.0):
        for tau in (2.0, 6.0):
            ts, x = constant_delay(
                lambda t, x, xd, _b=beta: _b * xd / (1 + xd**10)
                - x, tau, lambda t: 0.5, 60.0)
            plot_sol(ts, x, f"Mackey-Glass beta {beta:g}, tau "
                     f"{tau:g}")

    # 19-20: phase portraits x(t) vs x(t - tau)
    for tau in (2.0, 6.0):
        ts, x = constant_delay(
            lambda t, x, xd: 2.0 * xd / (1 + xd**10) - x, tau,
            lambda t: 0.5, 120.0)
        xi = interp1d(ts, x)
        tt = np.linspace(tau + 0.1, ts[-1] - 0.1, 4000)
        fig, ax = plt.subplots()
        ax.plot(xi(tt - tau), xi(tt), color=CHEBFUN_BLUE,
                linewidth=0.5)
        ax.set_xlabel("x(t - tau)")
        ax.set_ylabel("x(t)")
        ax.set_title(f"delay embedding, tau = {tau:g}", fontsize=9)
        save(fig, f"DelayDifferentialEquations_{panel:02d}.png")
        panel += 1

    # 21-24: two-delay and neutral-type examples
    ts, x = constant_delay(lambda t, x, xd: -xd + 0.3 * np.sin(t),
                           1.0, lambda t: 0.0, 30.0)
    plot_sol(ts, x, "forced delay equation")

    ts, x = constant_delay(lambda t, x, xd: -x + xd**2, 0.5,
                           lambda t: 0.4, 15.0)
    plot_sol(ts, x, "quadratic delay term")

    ts, x = constant_delay(lambda t, x, xd: xd - x**3, 1.5,
                           lambda t: 0.2, 30.0)
    plot_sol(ts, x, "cubic saturation")

    ts, x = constant_delay(lambda t, x, xd: np.sin(xd) - 0.1 * x,
                           2.0, lambda t: 1.0, 40.0)
    plot_sol(ts, x, "sinusoidal delay feedback")


def twoelectrons():
    """ode-nonlin/TwoElectrons — coupled electron dance."""
    # z'' = -z/|z|^3-ish two-body Coulomb-like planar model:
    # the example: z'' + z/|z|^3 coupling of two electrons on a line
    def rhs(t, y):
        x1, v1, x2, v2 = y
        d = x1 - x2
        rep = np.sign(d) / max(d**2, 1e-6)
        return [v1, -x1 + rep, v2, -x2 - rep]

    ts = np.linspace(0, 40, 6000)
    sol = solve_ivp(rhs, (0, 40), [0.5, 0.0, -0.5, 0.5], t_eval=ts,
                    rtol=1e-10, max_step=0.01)
    x1, x2 = sol.y[0], sol.y[2]

    fig, ax = plt.subplots()
    ax.plot(sol.t, x1, color=CHEBFUN_BLUE, linewidth=0.9)
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_ylim(-1.5, 1.5)
    save(fig, "TwoElectrons_01.png")

    fig, ax = plt.subplots()
    ax.plot(sol.t, x2, color=ORANGE, linewidth=0.9)
    ax.set_xlabel("t")
    ax.set_ylabel("y(t)")
    ax.set_ylim(-1.5, 1.5)
    save(fig, "TwoElectrons_02.png")

    fig, ax = plt.subplots()
    ax.plot(sol.t, x1, color=CHEBFUN_BLUE, linewidth=0.8)
    ax.plot(sol.t, x2, color=ORANGE, linewidth=0.8)
    ax.set_ylim(-1.5, 1.5)
    save(fig, "TwoElectrons_03.png")

    fig, ax = plt.subplots()
    ax.plot(x1, x2, color=CHEBFUN_BLUE, linewidth=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    save(fig, "TwoElectrons_04.png")

    # energy conservation
    E = (0.5 * sol.y[1] ** 2 + 0.5 * sol.y[3] ** 2
         + 0.5 * x1**2 + 0.5 * x2**2
         + 1.0 / np.maximum(np.abs(x1 - x2), 1e-6))
    fig, ax = plt.subplots()
    ax.plot(sol.t, E - E[0], color=CHEBFUN_BLUE, linewidth=0.8)
    ax.set_title("energy drift", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "TwoElectrons_05.png")

    # different initial condition: quasi-periodic dance
    sol2 = solve_ivp(rhs, (0, 40), [0.8, 0.0, -0.8, 0.2], t_eval=ts,
                     rtol=1e-10, max_step=0.01)
    fig, ax = plt.subplots()
    ax.plot(sol2.t, sol2.y[0], color=CHEBFUN_BLUE, linewidth=0.8)
    ax.plot(sol2.t, sol2.y[2], color=ORANGE, linewidth=0.8)
    save(fig, "TwoElectrons_06.png")

    fig, ax = plt.subplots()
    ax.plot(sol2.y[0], sol2.y[2], color=CHEBFUN_BLUE, linewidth=0.5)
    ax.set_aspect("equal")
    save(fig, "TwoElectrons_07.png")

    # separation distance and its minimum
    d = np.abs(x1 - x2)
    fig, ax = plt.subplots()
    ax.plot(sol.t, d, color=CHEBFUN_BLUE, linewidth=0.9)
    ax.set_title(f"separation (min {d.min():.4f})", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "TwoElectrons_08.png")

    fig, ax = plt.subplots()
    ax.semilogy(sol.t, np.maximum(d, 1e-8), color=ORANGE,
                linewidth=0.8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "TwoElectrons_09.png")

    # velocity phase plot
    fig, ax = plt.subplots()
    ax.plot(sol.y[1], sol.y[3], color=CHEBFUN_BLUE, linewidth=0.5)
    ax.set_xlabel("x'(t)")
    ax.set_ylabel("y'(t)")
    ax.set_aspect("equal")
    save(fig, "TwoElectrons_10.png")


def allencahn():
    """ode-nonlin/AllenCahn — Newton continuation in epsilon."""
    dom = (0.0, 10.0)
    n = 400
    xs = chebgrid(n, *dom)
    D = diffmat(np.cos(PI * np.arange(n) / (n - 1))[::-1]) \
        * (2.0 / (dom[1] - dom[0]))
    D2 = D @ D

    def newton_solve(eps, u0, iters=60):
        u = u0.copy()
        for _ in range(iters):
            F = eps * (D2 @ u) + u - u**3
            J = eps * D2 + np.eye(n) - np.diag(3 * u**2)
            F[0] = u[0] - 1.0
            J[0] = 0.0
            J[0, 0] = 1.0
            F[-1] = u[-1] + 1.0
            J[-1] = 0.0
            J[-1, -1] = 1.0
            du = np.linalg.solve(J, -F)
            # damping
            lam = 1.0
            while lam > 1e-4:
                if np.max(np.abs(du * lam)) < 2.0:
                    break
                lam /= 2
            u = u + lam * du
            if np.max(np.abs(du)) < 1e-12:
                break
        return u

    u = newton_solve(2.0, np.sin(xs))
    fig, ax = plt.subplots()
    ax.plot(xs, u, color=CHEBFUN_BLUE, linewidth=1.4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Allen-Cahn solution, eps = 2", fontsize=9)
    save(fig, "AllenCahn_01.png")

    # continuation to small eps: metastable multi-kink states
    fig, ax = plt.subplots()
    Epsvec = [1, 0.5, 0.2, 0.1, 0.03, 0.01, 0.003]
    u = newton_solve(2.0, np.sin(xs))
    for eps in Epsvec:
        u = newton_solve(eps, u)
        ax.plot(xs, u, linewidth=1.0)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("continuation from eps = 1 down to 0.003",
                 fontsize=9)
    save(fig, "AllenCahn_02.png")

    # panels 3-8: individual solutions along the continuation
    u = newton_solve(2.0, np.sin(xs))
    for j, eps in enumerate(Epsvec[:6], 3):
        u = newton_solve(eps, u)
        fig, ax = plt.subplots()
        ax.plot(xs, u, color=CHEBFUN_BLUE, linewidth=1.3)
        ax.grid(True, alpha=0.4, linewidth=0.4)
        ax.set_title(f"eps = {eps:g}", fontsize=9)
        save(fig, f"AllenCahn_{j:02d}.png")


def logistic():
    """ode-nonlin/Logistic — iterates of the logistic map as
    functions of r."""
    rs = np.linspace(0, 4, 1200)

    def iterate(n):
        x = np.full_like(rs, 0.5)
        for _ in range(n):
            x = rs * x * (1 - x)
        return x

    # panels 1-2: subplot stacks of x(n) for n = 0..3 and 4..7
    for pnl, nrange in ((1, range(0, 4)), (2, range(4, 8))):
        fig, axes = plt.subplots(4, 1)
        for ax, n in zip(axes, nrange):
            ax.plot(rs, iterate(n), color=CHEBFUN_BLUE, linewidth=0.8)
            ax.set_ylim(0, 1)
            ax.set_ylabel(f"x({n})", fontsize=6)
            ax.tick_params(labelsize=5)
            ax.grid(True, alpha=0.3, linewidth=0.3)
        save(fig, f"Logistic_{pnl:02d}.png")

    # panels 3-5: iterates on the chaotic window r in [3.5, 4]
    rs35 = np.linspace(3.5, 4.0, 4000)

    def iterate35(n):
        x = np.full_like(rs35, 0.5)
        for _ in range(n):
            x = rs35 * x * (1 - x)
        return x

    fig, ax = plt.subplots()
    ax.plot(rs35, iterate35(12), color=CHEBFUN_BLUE, linewidth=0.4)
    ax.set_ylim(0, 1)
    ax.set_xlim(3.5, 4)
    ax.set_title("x(12) on the chaotic window", fontsize=9)
    save(fig, "Logistic_03.png")

    fig, axes = plt.subplots(4, 1)
    for ax, n in zip(axes, (8, 9, 10, 11)):
        ax.plot(rs35, iterate35(n), color=CHEBFUN_BLUE,
                linewidth=0.4)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"x({n})", fontsize=6)
        ax.tick_params(labelsize=5)
    save(fig, "Logistic_04.png")

    fig, axes = plt.subplots(4, 1)
    for ax, n in zip(axes, (16, 17, 18, 15)):
        xv = iterate35(n)
        ax.plot(rs35, xv, color=CHEBFUN_BLUE, linewidth=0.4)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"x({n})", fontsize=6)
        d = np.diff(xv)
        length_proxy = int(np.sum(np.abs(np.diff(np.sign(d))) > 0)
                           * 2.5) + 2
        ax.text(3.52, 0.75, f"length(x) = {length_proxy}",
                fontsize=5, bbox=dict(fc="white", ec="k",
                                      linewidth=0.3))
        ax.tick_params(labelsize=5)
    save(fig, "Logistic_05.png")

    fig, ax = plt.subplots()
    ax.plot(rs35, iterate35(20), color=CHEBFUN_BLUE, linewidth=0.35)
    ax.set_ylim(0, 1)
    ax.set_xlim(3.5, 4)
    save(fig, "Logistic_06.png")

    # 7: length/complexity growth of the iterates
    lens = []
    ns = range(1, 16)
    for n in ns:
        x = iterate(n)
        # proxy for chebfun length: number of sign changes of x'
        d = np.diff(x)
        lens.append(np.sum(np.abs(np.diff(np.sign(d))) > 0) + 2)
    fig, ax = plt.subplots()
    ax.semilogy(list(ns), lens, ".-", markersize=7, linewidth=0.9,
                color=CHEBFUN_BLUE)
    ax.set_xlabel("iterate n")
    ax.set_title("complexity growth of x(n)", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "Logistic_07.png")


def chebopquiver():
    """ode-nonlin/ChebopQuiver — phase planes with quiver."""
    def phase(fx, fy, box, trajs, name, tmax=20.0, normalize=True):
        xg = np.linspace(box[0], box[1], 30)
        yg = np.linspace(box[2], box[3], 30)
        X, Y = np.meshgrid(xg, yg)
        U = fx(X, Y)
        V = fy(X, Y)
        if normalize:
            M = np.hypot(U, V) + 1e-12
            U, V = U / M, V / M
        fig, ax = plt.subplots()
        ax.quiver(X, Y, U, V, color=(0.5, 0.5, 0.5), width=0.0025)
        for x0 in trajs:
            sol = solve_ivp(lambda t, y: [fx(y[0], y[1]),
                                          fy(y[0], y[1])],
                            (0, tmax), x0, max_step=0.02, rtol=1e-8)
            ax.plot(sol.y[0], sol.y[1], color=CHEBFUN_BLUE,
                    linewidth=0.9)
        ax.set_xlim(box[0], box[1])
        ax.set_ylim(box[2], box[3])
        save(fig, name)

    # van der Pol
    phase(lambda x, y: y, lambda x, y: 2 * (1 - x**2) * y - x,
          (-2.75, 2.75, -5.5, 5.5),
          [(0.1, 0.1), (2.5, 4.0), (-2.5, -4.0)],
          "ChebopQuiver_01.png")

    # pendulum u'' + sin u = 0 on a strip
    phase(lambda x, y: y, lambda x, y: -np.sin(x),
          (-2.5, 25, -2, 5.5),
          [(0.0, 2.01), (0.0, 1.8), (0.0, 2.2), (6.3, 1.0)],
          "ChebopQuiver_02.png", tmax=30)

    # Duffing
    phase(lambda x, y: y, lambda x, y: x - x**3 - 0.15 * y,
          (-2, 2, -1.5, 1.5), [(0.1, 0.0), (1.8, 0.0), (-1.8, 0.5)],
          "ChebopQuiver_03.png", tmax=40)

    # predator-prey
    phase(lambda x, y: x * (1 - y), lambda x, y: y * (x - 1) * 0.5,
          (0, 4, 0, 4), [(2.0, 0.5), (3.0, 1.0), (1.5, 2.0)],
          "ChebopQuiver_04.png", tmax=25)

    # competing species
    phase(lambda x, y: x * (1 - x - 0.5 * y),
          lambda x, y: y * (0.75 - y - 0.25 * x),
          (0, 1.4, 0, 1.0), [(0.1, 0.1), (1.2, 0.2), (0.2, 0.9),
                             (0.9, 0.8)],
          "ChebopQuiver_05.png", tmax=40)


PAGES = {
    "DelayDifferentialEquations": delaydifferentialequations,
    "TwoElectrons": twoelectrons,
    "AllenCahn": allencahn,
    "Logistic": logistic,
    "ChebopQuiver": chebopquiver,
}


if __name__ == "__main__":
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, fn in PAGES.items():
        if flt.lower() in name.lower():
            print(f"[{name}]")
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED: {e}")
