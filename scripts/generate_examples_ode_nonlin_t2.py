"""Generate per-block figures for the ode-nonlin example category,
tranche 2: seventeen smaller pages.
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


def threeplanets():
    """ode-nonlin/ThreePlanets — animation frames on black."""
    rng = np.random.default_rng(2)
    y0c = np.concatenate([[0.0, 1.2, -0.9, -0.6, 0.9, -0.5],
                          rng.standard_normal(6) * 0.12])

    def rhs(t, y):
        p = y[:6].reshape(3, 2)
        v = y[6:].reshape(3, 2)
        a = np.zeros_like(p)
        for i in range(3):
            for j in range(3):
                if i != j:
                    d = p[j] - p[i]
                    a[i] += d / np.linalg.norm(d) ** 3
        return np.concatenate([v.ravel(), a.ravel()])

    T = 6.0
    sol = solve_ivp(rhs, (0, T), y0c,
                    t_eval=np.linspace(0, T, 400), rtol=1e-10,
                    atol=1e-10)
    P = sol.y[:6].reshape(3, 2, -1)
    colors = ["#00dd00", "red", "yellow"]
    for j, frac in enumerate((0.0, 1 / 3, 2 / 3, 1.0), 1):
        k = min(int(frac * (P.shape[2] - 1)), P.shape[2] - 1)
        fig = plt.figure()
        ax = fig.add_axes([0.06, 0.02, 0.88, 0.86])
        ax.set_facecolor("black")
        for i in range(3):
            ax.plot(P[i, 0, :k + 1], P[i, 1, :k + 1],
                    color=colors[i], linewidth=0.4, alpha=0.5)
            ax.plot([P[i, 0, k]], [P[i, 1, k]], ".",
                    color=colors[i], markersize=14)
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-1.8, 1.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t = {frac * T:g}", fontsize=12)
        save(fig, f"ThreePlanets_{j:02d}.png")


def picard():
    """ode-nonlin/Picard — Picard iteration for an IVP."""
    # y' = t y, y(0) = 1: Picard iterates converge to exp(t^2/2)
    ts = np.linspace(0, 1.5, 600)
    y = np.ones_like(ts)
    iterates = [y.copy()]
    for _ in range(8):
        integ = ts * y
        y = 1 + np.concatenate([[0], np.cumsum(
            0.5 * (integ[1:] + integ[:-1]) * np.diff(ts))])
        iterates.append(y.copy())

    fig, ax = plt.subplots()
    for it in iterates[:6]:
        ax.plot(ts, it, linewidth=1.0)
    ax.plot(ts, np.exp(ts**2 / 2), "k--", linewidth=1.2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Picard iterates converging to exp(t^2/2)",
                 fontsize=9)
    save(fig, "Picard_01.png")

    fig, ax = plt.subplots()
    errs = [np.max(np.abs(it - np.exp(ts**2 / 2)))
            for it in iterates]
    ax.semilogy(range(len(errs)), errs, ".-", markersize=8,
                linewidth=1.0, color=CHEBFUN_BLUE)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_xlabel("iteration")
    ax.set_title("Picard convergence", fontsize=9)
    save(fig, "Picard_02.png")

    # nonlinear example y' = y^2, y(0)=1 on [0, 0.9]: blow-up at 1
    ts2 = np.linspace(0, 0.9, 500)
    y = np.ones_like(ts2)
    its2 = [y.copy()]
    for _ in range(12):
        integ = y**2
        y = 1 + np.concatenate([[0], np.cumsum(
            0.5 * (integ[1:] + integ[:-1]) * np.diff(ts2))])
        its2.append(y.copy())
    fig, ax = plt.subplots()
    for it in its2[:8]:
        ax.plot(ts2, it, linewidth=0.9)
    ax.plot(ts2, 1 / (1 - ts2), "k--", linewidth=1.2)
    ax.set_ylim(0, 12)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("y' = y^2: iterates approach 1/(1-t)", fontsize=9)
    save(fig, "Picard_03.png")

    fig, ax = plt.subplots()
    errs2 = [np.max(np.abs(it - 1 / (1 - ts2))) for it in its2]
    ax.semilogy(range(len(errs2)), errs2, ".-", markersize=8,
                linewidth=1.0, color=ORANGE)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("slower convergence near blow-up", fontsize=9)
    save(fig, "Picard_04.png")


def orbits():
    """ode-nonlin/Orbits — Kepler orbits at several eccentricities."""
    def rhs(t, y):
        r = np.hypot(y[0], y[1])
        return [y[2], y[3], -y[0] / r**3, -y[1] / r**3]

    fig, ax = plt.subplots()
    for e, c in ((0.0, "b"), (0.3, "r"), (0.6, "g"), (0.8, "m")):
        v0 = np.sqrt(1 + e)
        sol = solve_ivp(rhs, (0, 20), [1 - 0 * e, 0, 0, v0],
                        t_eval=np.linspace(0, 20, 3000), rtol=1e-11)
        ax.plot(sol.y[0], sol.y[1], c, linewidth=0.9)
    ax.plot([0], [0], ".k", markersize=12)
    ax.set_aspect("equal")
    ax.set_title("Kepler orbits, e = 0, 0.3, 0.6, 0.8", fontsize=9)
    save(fig, "Orbits_01.png")

    # energy and angular momentum conservation for e = 0.6
    sol = solve_ivp(rhs, (0, 60), [1, 0, 0, np.sqrt(1.6)],
                    t_eval=np.linspace(0, 60, 8000), rtol=1e-11)
    r = np.hypot(sol.y[0], sol.y[1])
    E = 0.5 * (sol.y[2] ** 2 + sol.y[3] ** 2) - 1 / r
    Lz = sol.y[0] * sol.y[3] - sol.y[1] * sol.y[2]
    fig, ax = plt.subplots()
    ax.plot(sol.t, E - E[0], linewidth=0.9, label="energy drift")
    ax.plot(sol.t, Lz - Lz[0], linewidth=0.9,
            label="ang. momentum drift")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "Orbits_02.png")

    fig, ax = plt.subplots()
    ax.plot(sol.t, r, color=CHEBFUN_BLUE, linewidth=0.9)
    ax.set_xlabel("t")
    ax.set_title("radius r(t): Kepler oscillation", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "Orbits_03.png")

    # perturbed potential: precessing orbit
    def rhs_p(t, y):
        r = np.hypot(y[0], y[1])
        f = -1 / r**3 - 0.02 / r**4
        return [y[2], y[3], f * y[0], f * y[1]]

    solp = solve_ivp(rhs_p, (0, 60), [1, 0, 0, 1.2],
                     t_eval=np.linspace(0, 60, 8000), rtol=1e-11)
    fig, ax = plt.subplots()
    ax.plot(solp.y[0], solp.y[1], color=CHEBFUN_BLUE, linewidth=0.5)
    ax.plot([0], [0], ".k", markersize=10)
    ax.set_aspect("equal")
    ax.set_title("precessing orbit in a perturbed potential",
                 fontsize=9)
    save(fig, "Orbits_04.png")


def lorenzattractor():
    """ode-nonlin/LorenzAttractor."""
    def rhs(t, y):
        return [10 * (y[1] - y[0]), 28 * y[0] - y[1] - y[0] * y[2],
                y[0] * y[1] - 8 / 3 * y[2]]

    sol = solve_ivp(rhs, (0, 50), [-14, -15, 20],
                    t_eval=np.linspace(0, 50, 20000), rtol=1e-10)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot3D(sol.y[0], sol.y[1], sol.y[2], color=CHEBFUN_BLUE,
              linewidth=0.3)
    ax.set_title("the Lorenz attractor", fontsize=9)
    save(fig, "LorenzAttractor_01.png")

    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[0], color=CHEBFUN_BLUE, linewidth=0.4)
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    save(fig, "LorenzAttractor_02.png")

    fig, ax = plt.subplots()
    ax.plot(sol.y[0], sol.y[2], color=CHEBFUN_BLUE, linewidth=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    save(fig, "LorenzAttractor_03.png")

    # sensitive dependence
    sol2 = solve_ivp(rhs, (0, 50), [-14, -15, 20 + 1e-9],
                     t_eval=np.linspace(0, 50, 20000), rtol=1e-10)
    fig, ax = plt.subplots()
    ax.semilogy(sol.t, np.abs(sol.y[0] - sol2.y[0]) + 1e-16,
                color=ORANGE, linewidth=0.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("sensitive dependence: 1e-9 perturbation grows",
                 fontsize=9)
    save(fig, "LorenzAttractor_04.png")


def logistic2():
    """ode-nonlin/Logistic2 — the continuous logistic equation."""
    ts = np.linspace(0, 10, 600)
    fig, ax = plt.subplots()
    for x0 in (0.02, 0.1, 0.5, 0.9, 1.4, 1.8):
        sol = solve_ivp(lambda t, y: [y[0] * (1 - y[0])], (0, 10),
                        [x0], t_eval=ts, rtol=1e-10)
        ax.plot(sol.t, sol.y[0], linewidth=1.1)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("logistic trajectories", fontsize=9)
    save(fig, "Logistic2_01.png")

    # harvested logistic: saddle-node at h = 1/4
    fig, ax = plt.subplots()
    for h, c in ((0.15, "b"), (0.24, "g"), (0.26, "r")):
        for x0 in (0.15, 0.5, 0.95):
            sol = solve_ivp(lambda t, y, _h=h: [y[0] * (1 - y[0])
                                                - _h],
                            (0, 20), [x0],
                            t_eval=np.linspace(0, 20, 600),
                            rtol=1e-9)
            ax.plot(sol.t, np.maximum(sol.y[0], -0.1), c,
                    linewidth=0.8)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("harvested logistic: h = 0.15, 0.24, 0.26",
                 fontsize=9)
    save(fig, "Logistic2_02.png")

    # bifurcation diagram of equilibria vs h
    hs = np.linspace(0, 0.35, 300)
    fig, ax = plt.subplots()
    disc = 1 - 4 * hs
    m = disc >= 0
    ax.plot(hs[m], (1 + np.sqrt(disc[m])) / 2, "b", linewidth=1.4)
    ax.plot(hs[m], (1 - np.sqrt(disc[m])) / 2, "r--", linewidth=1.4)
    ax.axvline(0.25, color="k", linewidth=0.6, linestyle=":")
    ax.set_xlabel("harvest h")
    ax.set_title("saddle-node bifurcation at h = 1/4", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "Logistic2_03.png")

    # time to extinction past the bifurcation
    hs2 = np.linspace(0.26, 0.5, 30)
    tex = []
    for h in hs2:
        sol = solve_ivp(lambda t, y, _h=h: [y[0] * (1 - y[0]) - _h],
                        (0, 200), [0.5], rtol=1e-9, dense_output=True)
        cross = sol.t[np.argmax(sol.y[0] < 0)] if np.any(
            sol.y[0] < 0) else np.nan
        tex.append(cross)
    fig, ax = plt.subplots()
    ax.plot(hs2, tex, ".-", markersize=6, linewidth=0.9,
            color=CHEBFUN_BLUE)
    ax.set_xlabel("h")
    ax.set_title("time to extinction ~ (h - 1/4)^{-1/2}", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "Logistic2_04.png")


def ivpcapabilities():
    """ode-nonlin/IVPCapabilities — a tour of IVP solves."""
    # van der Pol
    sol = solve_ivp(lambda t, y: [y[1], 10 * (1 - y[0] ** 2) * y[1]
                                  - y[0]],
                    (0, 60), [2, 0], t_eval=np.linspace(0, 60, 6000),
                    rtol=1e-9)
    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[0], color=CHEBFUN_BLUE, linewidth=0.9)
    ax.set_title("van der Pol, mu = 10", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "IVPCapabilities_01.png")

    # stiff problem: Robertson-like scaled
    sol2 = solve_ivp(lambda t, y: [-0.04 * y[0] + 1e4 * y[1] * y[2],
                                   0.04 * y[0] - 1e4 * y[1] * y[2]
                                   - 3e7 * y[1] ** 2,
                                   3e7 * y[1] ** 2],
                     (1e-6, 1e4), [1, 0, 0], method="BDF",
                     rtol=1e-8, atol=1e-12)
    fig, ax = plt.subplots()
    ax.semilogx(sol2.t, sol2.y[0], linewidth=1.1, label="A")
    ax.semilogx(sol2.t, 1e4 * sol2.y[1], linewidth=1.1,
                label="1e4 B")
    ax.semilogx(sol2.t, sol2.y[2], linewidth=1.1, label="C")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Robertson stiff kinetics", fontsize=9)
    save(fig, "IVPCapabilities_02.png")

    # event detection: bouncing ball
    def hit(t, y):
        return y[0]

    hit.terminal = True
    hit.direction = -1
    ts_all, xs_all = [], []
    t0, y0v = 0.0, [1.0, 0.0]
    for _ in range(8):
        sol3 = solve_ivp(lambda t, y: [y[1], -9.8], (t0, t0 + 5),
                         y0v, events=hit, max_step=0.01)
        ts_all.append(sol3.t)
        xs_all.append(sol3.y[0])
        if not sol3.t_events[0].size:
            break
        t0 = sol3.t_events[0][0]
        y0v = [0.0, -0.75 * sol3.y_events[0][0][1]]
    fig, ax = plt.subplots()
    ax.plot(np.concatenate(ts_all), np.concatenate(xs_all),
            color=CHEBFUN_BLUE, linewidth=1.1)
    ax.set_title("bouncing ball via event detection", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "IVPCapabilities_03.png")

    # high-accuracy long integration: energy of an oscillator
    sol4 = solve_ivp(lambda t, y: [y[1], -np.sin(y[0])], (0, 100),
                     [2.0, 0.0], t_eval=np.linspace(0, 100, 8000),
                     rtol=1e-12, atol=1e-12)
    E = 0.5 * sol4.y[1] ** 2 - np.cos(sol4.y[0])
    fig, ax = plt.subplots()
    ax.plot(sol4.t, E - E[0], color=ORANGE, linewidth=0.8)
    ax.set_title("pendulum energy drift at rtol 1e-12", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "IVPCapabilities_04.png")


def guckenheimerholmes():
    """ode-nonlin/GuckenheimerHolmes — the GH limit-cycle system."""
    def rhs(t, y):
        x, yy, z = y
        return [x * (1 - x**2 - yy**2 - z**2) - yy * (1 + z),
                yy * (1 - x**2 - yy**2 - z**2) + x * (1 + z),
                -z * (x**2 + yy**2)]

    sol = solve_ivp(rhs, (0, 100), [0.1, 0.1, 0.8],
                    t_eval=np.linspace(0, 100, 12000), rtol=1e-10)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot3D(sol.y[0], sol.y[1], sol.y[2], color=CHEBFUN_BLUE,
              linewidth=0.4)
    save(fig, "GuckenheimerHolmes_01.png")

    fig, ax = plt.subplots()
    for j, c in ((0, "b"), (1, "r"), (2, "g")):
        ax.plot(sol.t, sol.y[j], c, linewidth=0.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "GuckenheimerHolmes_02.png")

    fig, ax = plt.subplots()
    ax.plot(sol.y[0], sol.y[1], color=CHEBFUN_BLUE, linewidth=0.4)
    ax.set_aspect("equal")
    save(fig, "GuckenheimerHolmes_03.png")

    r = np.hypot(sol.y[0], sol.y[1])
    fig, ax = plt.subplots()
    ax.plot(sol.t, r, color=ORANGE, linewidth=0.7)
    ax.plot(sol.t, sol.y[2], color=CHEBFUN_BLUE, linewidth=0.7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("radius and z: spiraling onto the cycle",
                 fontsize=9)
    save(fig, "GuckenheimerHolmes_04.png")


def exactsolns():
    """ode-nonlin/ExactSolns — ODEs with closed forms."""
    ts = np.linspace(0.01, 3, 600)
    # y' = y^2 - solution 1/(1-t) vs numeric
    sol = solve_ivp(lambda t, y: [y[0] ** 2], (0.01, 0.9),
                    [1 / 0.99], t_eval=np.linspace(0.01, 0.9, 300),
                    rtol=1e-11)
    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[0], color=CHEBFUN_BLUE, linewidth=1.4)
    ax.plot(sol.t, 1 / (1 - sol.t), "k--", linewidth=1.0)
    ax.set_title("y' = y^2 vs 1/(1-t)", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "ExactSolns_01.png")

    # Bernoulli: y' + y = y^3
    sol2 = solve_ivp(lambda t, y: [-y[0] + y[0] ** 3], (0, 3),
                     [0.5], t_eval=ts, rtol=1e-11)
    exact = 1 / np.sqrt(3 * np.exp(2 * ts) + 1)
    fig, ax = plt.subplots()
    ax.plot(ts, sol2.y[0], color=CHEBFUN_BLUE, linewidth=1.4)
    ax.plot(ts, exact, "k--", linewidth=1.0)
    ax.set_title("Bernoulli equation vs closed form", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "ExactSolns_02.png")

    # separable: y' = t/y
    sol3 = solve_ivp(lambda t, y: [t / y[0]], (0, 3), [1.0],
                     t_eval=ts, rtol=1e-11)
    fig, ax = plt.subplots()
    ax.plot(ts, sol3.y[0], color=CHEBFUN_BLUE, linewidth=1.4)
    ax.plot(ts, np.sqrt(1 + ts**2), "k--", linewidth=1.0)
    ax.set_title("y' = t/y vs sqrt(1+t^2)", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "ExactSolns_03.png")

    # errors
    fig, ax = plt.subplots()
    ax.semilogy(ts, np.maximum(np.abs(sol2.y[0] - exact), 1e-18),
                linewidth=0.9, label="Bernoulli")
    ax.semilogy(ts, np.maximum(np.abs(sol3.y[0]
                                      - np.sqrt(1 + ts**2)), 1e-18),
                linewidth=0.9, label="separable")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "ExactSolns_04.png")


def squarecycle():
    """ode-nonlin/SquareCycle — a nearly square limit cycle."""
    def rhs(t, y):
        x, yy = y
        return [yy**3 - x**7 * 0 + yy * (1 - x**8),
                -x + 0.1 * yy * (1 - x**2)] if False else \
            [yy, -x**7 + 3 * yy * (1 - x**8)]

    sol = solve_ivp(rhs, (0, 80), [0.1, 0.1],
                    t_eval=np.linspace(0, 80, 12000), rtol=1e-10)
    fig, ax = plt.subplots()
    ax.plot(sol.y[0], sol.y[1], color=CHEBFUN_BLUE, linewidth=0.6)
    ax.set_aspect("equal")
    ax.set_title("a squarish limit cycle", fontsize=9)
    save(fig, "SquareCycle_01.png")

    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[0], color=CHEBFUN_BLUE, linewidth=0.7)
    ax.set_xlabel("t")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "SquareCycle_02.png")

    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[1], color=ORANGE, linewidth=0.7)
    ax.set_xlabel("t")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "SquareCycle_03.png")


def modellingdiseases():
    """ode-nonlin/ModellingDiseases — SIR dynamics."""
    def sir(beta, gamma):
        return solve_ivp(
            lambda t, y: [-beta * y[0] * y[1],
                          beta * y[0] * y[1] - gamma * y[1],
                          gamma * y[1]],
            (0, 100), [0.99, 0.01, 0.0],
            t_eval=np.linspace(0, 100, 2000), rtol=1e-10)

    sol = sir(0.4, 0.1)
    fig, ax = plt.subplots()
    for j, (c, lbl) in enumerate((("b", "S"), ("r", "I"),
                                  ("g", "R"))):
        ax.plot(sol.t, sol.y[j], c, linewidth=1.2, label=lbl)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("SIR epidemic, R0 = 4", fontsize=9)
    save(fig, "ModellingDiseases_01.png")

    # varying R0
    fig, ax = plt.subplots()
    for beta in (0.15, 0.25, 0.4, 0.6):
        sol = sir(beta, 0.1)
        ax.plot(sol.t, sol.y[1], linewidth=1.0,
                label=f"R0 = {beta/0.1:g}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("infected curves for several R0", fontsize=9)
    save(fig, "ModellingDiseases_02.png")

    # phase plane S-I
    fig, ax = plt.subplots()
    for i0 in (0.01, 0.05, 0.1, 0.2):
        sol = solve_ivp(
            lambda t, y: [-0.4 * y[0] * y[1],
                          0.4 * y[0] * y[1] - 0.1 * y[1]],
            (0, 200), [1 - i0, i0],
            t_eval=np.linspace(0, 200, 2000), rtol=1e-10)
        ax.plot(sol.y[0], sol.y[1], linewidth=1.0)
    ax.set_xlabel("S")
    ax.set_ylabel("I")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "ModellingDiseases_03.png")


def gulfstream():
    """ode-nonlin/GulfStream — streamlines of a jet flow."""
    # a Bickley-jet-like streamfunction with meanders
    xg = np.linspace(0, 10, 300)
    yg = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(xg, yg)
    psi = -np.tanh(Y - 0.6 * np.sin(0.8 * X)) + 0.1 * np.sin(Y)

    fig, ax = plt.subplots()
    cs = ax.contour(X, Y, psi, levels=21, linewidths=0.7,
                    colors="k")
    ax.set_title("streamlines of a meandering jet", fontsize=9)
    save(fig, "GulfStream_01.png")

    # particle trajectories
    def vel(t, y):
        x, yy = y
        dpsidy = (-1 / np.cosh(yy - 0.6 * np.sin(0.8 * x)) ** 2
                  + 0.1 * np.cos(yy))
        dpsidx = (1 / np.cosh(yy - 0.6 * np.sin(0.8 * x)) ** 2
                  * 0.48 * np.cos(0.8 * x))
        return [-dpsidy, dpsidx]

    fig, ax = plt.subplots()
    ax.contour(X, Y, psi, levels=15, linewidths=0.4,
               colors=[(0.8, 0.8, 0.8)])
    for y0 in (-0.5, 0.0, 0.5, 1.0):
        sol = solve_ivp(vel, (0, 20), [0.0, y0],
                        t_eval=np.linspace(0, 20, 2000), rtol=1e-9)
        ax.plot(sol.y[0], sol.y[1], linewidth=1.0)
    ax.set_xlim(0, 10)
    ax.set_ylim(-3, 3)
    ax.set_title("drifter trajectories", fontsize=9)
    save(fig, "GulfStream_02.png")

    fig, ax = plt.subplots()
    U = -np.gradient(psi, yg, axis=0)
    ax.plot(U[:, 0], yg, color=CHEBFUN_BLUE, linewidth=1.4)
    ax.set_xlabel("u(y)")
    ax.set_ylabel("y")
    ax.set_title("jet velocity profile", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "GulfStream_03.png")


def droplets():
    """ode-nonlin/Droplets — sessile drops (Young-Laplace)."""
    def drop_profile(kappa0, L=12.0):
        def rhs(s, y):
            r, z, phi = y
            k = kappa0 - z
            dphi = k - np.sin(phi) / max(r, 1e-8)
            return [np.cos(phi), np.sin(phi), dphi]

        sol = solve_ivp(rhs, (1e-8, L), [1e-8, 0.0, 0.0],
                        t_eval=np.linspace(1e-8, L, 2400),
                        rtol=1e-10)
        # cut where the tangent angle reaches pi (no wetting)
        phi = sol.y[2]
        stop = np.argmax(phi >= PI) if np.any(phi >= PI) else len(phi)
        return (sol.y[0][:stop], sol.y[1][:stop])

    def sessile_plot(kappa0, title, name):
        r, z = drop_profile(kappa0)
        zmax = z[-1]
        # flip so the drop sits on the ground line z = 0
        zz = zmax - z
        fig, ax = plt.subplots()
        xs_ = np.concatenate([-r[::-1], r])
        ys_ = np.concatenate([zz[::-1], zz])
        ax.fill(xs_, ys_, color=(0.35, 0.45, 0.95))
        ax.plot(xs_, ys_, "k", linewidth=1.6)
        ax.axhline(0, color="k", linewidth=1.2)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-0.8, 2.8)
        ax.set_title(title, fontsize=10)
        save(fig, name)

    sessile_plot(1.2, "small drop", "Droplets_01.png")
    sessile_plot(0.75, "no wetting", "Droplets_02.png")
    sessile_plot(0.5, "a large flattened drop", "Droplets_03.png")


def carrier():
    """ode-nonlin/Carrier — the Carrier equation multi-solutions."""
    dom = (-1.0, 1.0)
    n = 400
    xs = chebgrid(n, *dom)
    D = diffmat(np.cos(PI * np.arange(n) / (n - 1))[::-1])
    D2 = D @ D
    eps = 0.01

    def newton(u0, iters=80):
        u = u0.copy()
        for _ in range(iters):
            F = eps * (D2 @ u) + 2 * (1 - xs**2) * u + u**2 - 1
            J = eps * D2 + np.diag(2 * (1 - xs**2) + 2 * u)
            F[0] = u[0]
            J[0] = 0.0
            J[0, 0] = 1.0
            F[-1] = u[-1]
            J[-1] = 0.0
            J[-1, -1] = 1.0
            du = np.linalg.solve(J, -F)
            lam = 1.0
            while np.max(np.abs(lam * du)) > 1.0 and lam > 1e-3:
                lam /= 2
            u = u + lam * du
            if np.max(np.abs(du)) < 1e-12:
                break
        return u

    u1 = newton(2 * (xs**2 - 1) * (1 - 2 / (1 + 20 * xs**2)))
    fig, ax = plt.subplots()
    ax.plot(xs, u1, color=CHEBFUN_BLUE, linewidth=1.3)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("a Carrier-equation solution", fontsize=9)
    save(fig, "Carrier_01.png")

    u2 = newton(np.sin(PI * xs) * 1.5)
    fig, ax = plt.subplots()
    ax.plot(xs, u2, color=ORANGE, linewidth=1.3)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("another solution branch", fontsize=9)
    save(fig, "Carrier_02.png")

    u3 = newton(np.sin(2 * PI * xs) * 1.5)
    fig, ax = plt.subplots()
    ax.plot(xs, u1, linewidth=1.0)
    ax.plot(xs, u2, linewidth=1.0)
    ax.plot(xs, u3, linewidth=1.0)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("multiple solutions of the Carrier equation",
                 fontsize=9)
    save(fig, "Carrier_03.png")


def blasius():
    """ode-nonlin/Blasius — the boundary-layer profile."""
    from scipy.optimize import brentq

    def shoot(s):
        sol = solve_ivp(lambda t, y: [y[1], y[2], -y[0] * y[2] / 2],
                        (0, 10), [0, 0, s], rtol=1e-11)
        return sol.y[1][-1] - 1.0

    s0 = brentq(shoot, 0.1, 1.0, xtol=1e-12)
    sol = solve_ivp(lambda t, y: [y[1], y[2], -y[0] * y[2] / 2],
                    (0, 10), [0, 0, s0],
                    t_eval=np.linspace(0, 10, 1000), rtol=1e-11)
    print(f"    Blasius f''(0) = {s0:.10f} (exact 0.3320573362...)")

    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[0], color=CHEBFUN_BLUE, linewidth=1.4)
    ax.set_xlabel("eta")
    ax.set_title("Blasius f(eta)", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "Blasius_01.png")

    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[1], color=ORANGE, linewidth=1.4)
    ax.set_xlabel("eta")
    ax.set_title("velocity profile f'(eta)", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "Blasius_02.png")

    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[2], color=CHEBFUN_BLUE, linewidth=1.4)
    ax.set_xlabel("eta")
    ax.set_title("shear f''(eta)", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "Blasius_03.png")


def lyapunovexponents():
    """ode-nonlin/LyapunovExponents — leading exponent of Lorenz."""
    def rhs(t, y):
        return [10 * (y[1] - y[0]), 28 * y[0] - y[1] - y[0] * y[2],
                y[0] * y[1] - 8 / 3 * y[2]]

    # Benettin: evolve two nearby trajectories with renormalization
    y1 = np.array([-14.0, -15.0, 20.0])
    d0 = 1e-8
    y2 = y1 + np.array([d0, 0, 0])
    T, nsteps = 0.5, 80
    lams = []
    running = []
    for k in range(nsteps):
        s1 = solve_ivp(rhs, (0, T), y1, rtol=1e-10)
        s2 = solve_ivp(rhs, (0, T), y2, rtol=1e-10)
        y1 = s1.y[:, -1]
        y2n = s2.y[:, -1]
        d = np.linalg.norm(y2n - y1)
        lams.append(np.log(d / d0) / T)
        running.append(np.mean(lams))
        y2 = y1 + (y2n - y1) * (d0 / d)
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, nsteps + 1) * T, running, ".-",
            markersize=5, linewidth=0.8, color=CHEBFUN_BLUE)
    ax.axhline(0.9056, color="r", linewidth=0.7, linestyle="--")
    ax.set_xlabel("time")
    ax.set_title(f"running Lyapunov estimate -> {running[-1]:.3f} "
                 "(ref 0.906)", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "LyapunovExponents_01.png")

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, nsteps + 1) * T, lams, ".", markersize=5,
            color=ORANGE)
    ax.set_xlabel("time")
    ax.set_title("local expansion rates", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "LyapunovExponents_02.png")


def fouriercollocationnonlin():
    """ode-nonlin/FourierCollocationNonLin — periodic nonlinear ODE."""
    # u' + u^3 = cos t periodic solution via Newton on Fourier grid
    n = 128
    xg = np.linspace(0, 2 * PI, n, endpoint=False)
    dx = xg[1] - xg[0]
    col = np.zeros(n)
    kk = np.arange(1, n)
    col[1:] = 0.5 * (-1.0) ** kk / np.tan(kk * dx / 2)
    Dp = -np.column_stack([np.roll(np.concatenate([[0.0], col[1:]]),
                                   k) for k in range(n)])
    u = 0.5 * np.cos(xg)
    for _ in range(40):
        F = Dp @ u + u**3 - np.cos(xg)
        J = Dp + np.diag(3 * u**2)
        du = np.linalg.solve(J, -F)
        u = u + du
        if np.max(np.abs(du)) < 1e-13:
            break
    fig, ax = plt.subplots()
    ax.plot(xg, u, color=CHEBFUN_BLUE, linewidth=1.4)
    ax.set_xlim(0, 2 * PI)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("periodic solution of u' + u^3 = cos t", fontsize=9)
    save(fig, "FourierCollocationNonLin_01.png")

    fig, ax = plt.subplots()
    resid = Dp @ u + u**3 - np.cos(xg)
    ax.semilogy(xg, np.maximum(np.abs(resid), 1e-18), "k",
                linewidth=0.8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("residual of the Newton solution", fontsize=9)
    save(fig, "FourierCollocationNonLin_02.png")


def bvpsystem():
    """ode-nonlin/BVPSystem — a coupled nonlinear BVP."""
    n = 300
    xs = chebgrid(n, 0.0, 1.0)
    D = diffmat(np.cos(PI * np.arange(n) / (n - 1))[::-1]) * 2.0
    D2 = D @ D

    # u'' = v u, v'' = u^2 - 1 with u(0)=1, u(1)=0, v(0)=0, v(1)=1
    u = 1 - xs
    v = xs.copy()
    for _ in range(60):
        Fu = D2 @ u - v * u
        Fv = D2 @ v - u**2 + 1
        J = np.block([[D2 - np.diag(v), -np.diag(u)],
                      [-np.diag(2 * u), D2]])
        F = np.concatenate([Fu, Fv])
        # BC rows
        for pos, val_idx in ((0, 0), (n - 1, n - 1), (n, n),
                             (2 * n - 1, 2 * n - 1)):
            J[pos] = 0.0
            J[pos, val_idx] = 1.0
        F[0] = u[0] - 1
        F[n - 1] = u[-1]
        F[n] = v[0]
        F[2 * n - 1] = v[-1] - 1
        duv = np.linalg.solve(J, -F)
        u = u + duv[:n]
        v = v + duv[n:]
        if np.max(np.abs(duv)) < 1e-12:
            break

    fig, ax = plt.subplots()
    ax.plot(xs, u, linewidth=1.4, label="u")
    ax.plot(xs, v, linewidth=1.4, label="v")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("coupled nonlinear BVP by Newton", fontsize=9)
    save(fig, "BVPSystem_01.png")

    fig, ax = plt.subplots()
    resid = np.abs(D2 @ u - v * u)
    ax.semilogy(xs, np.maximum(resid, 1e-18), "k", linewidth=0.8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("residual of the first equation", fontsize=9)
    save(fig, "BVPSystem_02.png")


PAGES = {
    "ThreePlanets": threeplanets,
    "Picard": picard,
    "Orbits": orbits,
    "LorenzAttractor": lorenzattractor,
    "Logistic2": logistic2,
    "IVPCapabilities": ivpcapabilities,
    "GuckenheimerHolmes": guckenheimerholmes,
    "ExactSolns": exactsolns,
    "SquareCycle": squarecycle,
    "ModellingDiseases": modellingdiseases,
    "GulfStream": gulfstream,
    "Droplets": droplets,
    "Carrier": carrier,
    "Blasius": blasius,
    "LyapunovExponents": lyapunovexponents,
    "FourierCollocationNonLin": fouriercollocationnonlin,
    "BVPSystem": bvpsystem,
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
