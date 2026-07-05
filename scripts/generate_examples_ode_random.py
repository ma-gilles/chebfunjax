"""Generate per-block figures for the ode-random example category.

Random forcing uses band-limited randnfun equivalents (fixed seeds,
statistical texture equivalence — the honest convention for randomized
demos); IVPs are integrated with solve_ivp (chebop IVP routing is
task #24).
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

    ref_path = os.path.join(REFROOT, "ode-random", name)
    size = Image.open(ref_path).size if os.path.exists(ref_path) else (600, 270)
    save_chebfun_figure(fig, os.path.join(DOCS, "ode-random", name),
                        size=size)
    plt.close(fig)
    print(f"  ode-random/{name} saved")


def randnfun_np(lam, dom, seed, normalized=True):
    """Band-limited random function: wavelength lam on dom.

    Returns a callable. 'norm' scaling gives white-noise-like
    normalization 1/sqrt(lam) as in MATLAB randnfun(...,'norm')."""
    rng = np.random.default_rng(seed)
    a, b = dom
    L = b - a
    m = int(2 * L / lam) + 1
    coefs = rng.standard_normal((m + 1, 2))

    def f(t):
        t = np.asarray(t, dtype=float)
        s = 2 * PI * (t - a) / L
        out = np.zeros_like(t)
        for k in range(m + 1):
            out = out + (coefs[k, 0] * np.cos(k * s)
                         + coefs[k, 1] * np.sin(k * s))
        out = out / np.sqrt(m + 1)
        if normalized:
            out = out / np.sqrt(lam)
        return out

    return f


def consensus():
    """ode-random/Consensus — two agents pulled together by coupling."""
    dom = (0.0, 40.0)
    f = randnfun_np(0.2, dom, 3)
    g = randnfun_np(0.2, dom, 4)
    ts = np.linspace(*dom, 4000)

    fig, ax = plt.subplots()
    ax.plot(ts, f(ts), color=CHEBFUN_BLUE, linewidth=0.6)
    ax.plot(ts, g(ts), color=ORANGE, linewidth=0.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("two random forcings", fontsize=10)
    save(fig, "Consensus_01.png")

    def rhs(F):
        def _rhs(t, y):
            u, v = y
            c = F * (u - v) * np.exp(-((u - v) ** 2))
            return [-f(t) - c, -g(t) + c]
        return _rhs

    for j, F in enumerate((0.0, 3.0), 2):
        sol = solve_ivp(rhs(F), dom, [0.0, 0.0], t_eval=ts,
                        max_step=0.02, rtol=1e-7)
        fig, ax = plt.subplots()
        ax.plot(sol.t, sol.y[0], color=CHEBFUN_BLUE, linewidth=0.8)
        ax.plot(sol.t, sol.y[1], color=ORANGE, linewidth=0.8)
        ax.grid(True, alpha=0.4, linewidth=0.4)
        ax.set_title(f"coupling F = {F:g}", fontsize=10)
        ax.set_xlabel("t")
        save(fig, f"Consensus_{j:02d}.png")


def gbm():
    """ode-random/GBM — geometric Brownian motion paths."""
    dom = (0.0, 20.0)
    ts = np.linspace(*dom, 4000)
    mu, sigma = 0.2, 0.4

    def paths(mu_):
        out = []
        for k in range(5):
            f = randnfun_np(0.2, dom, 10 + k)
            sol = solve_ivp(
                lambda t, y: [mu_ * y[0] + sigma * f(t) * y[0]],
                dom, [1.0], t_eval=ts, max_step=0.02, rtol=1e-7)
            out.append(np.clip(sol.y[0], -100, 100))
        return out

    fig, ax = plt.subplots()
    f0 = randnfun_np(0.2, dom, 10)
    ax.plot(ts, f0(ts), color=CHEBFUN_BLUE, linewidth=0.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("white-noise-like forcing", fontsize=10)
    save(fig, "GBM_01.png")

    for j, mu_ in enumerate((0.2, 0.0), 2):
        fig, ax = plt.subplots()
        for y in paths(mu_):
            ax.plot(ts, y, linewidth=0.8)
        ax.grid(True, alpha=0.4, linewidth=0.4)
        ax.set_title(f"five GBM paths, mu = {mu_:g}", fontsize=10)
        ax.set_xlabel("t")
        save(fig, f"GBM_{j:02d}.png")


def levelhopping():
    """ode-random/LevelHopping — bistable well with random kicks."""
    dom = (0.0, 100.0)
    ts = np.linspace(*dom, 8000)

    for j, lam in enumerate((0.4, 0.2), 1):
        f = randnfun_np(lam, dom, 0)
        sol = solve_ivp(
            lambda t, y: [y[0] - y[0] ** 3 + 0.7 * f(t)],
            dom, [0.0], t_eval=ts, max_step=0.02, rtol=1e-7)
        fig, ax = plt.subplots()
        ax.plot(sol.t, sol.y[0], color=CHEBFUN_BLUE, linewidth=0.7)
        ax.grid(True, alpha=0.4, linewidth=0.4)
        ax.set_xlabel("t")
        ax.set_title(f"level hopping, lambda = {lam:g}", fontsize=10)
        save(fig, f"LevelHopping_{j:02d}.png")


def phaselocking():
    """ode-random/PhaseLocking — noisy coupled oscillators."""
    dom = (0.0, 60.0)
    ts = np.linspace(*dom, 6000)

    f = randnfun_np(0.5, dom, 1, normalized=True)

    for j, K in enumerate((0.2, 1.2), 1):
        def rhs(t, y):
            th1, th2 = y
            return [1.0 + K * np.sin(th2 - th1) + 0.3 * f(t),
                    1.3 + K * np.sin(th1 - th2)]

        sol = solve_ivp(rhs, dom, [0.0, PI / 2], t_eval=ts,
                        max_step=0.05, rtol=1e-8)
        fig, ax = plt.subplots()
        ax.plot(sol.t, np.sin(sol.y[0] - sol.y[1]),
                color=CHEBFUN_BLUE, linewidth=0.7)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.4, linewidth=0.4)
        ax.set_title(f"sin(phase difference), K = {K:g}", fontsize=10)
        ax.set_xlabel("t")
        save(fig, f"PhaseLocking_{j:02d}.png")

    # lambda = 0.05, 60 paths: ensemble splitting into two branches
    dom6 = (0.0, 6.0)
    ts6 = np.linspace(*dom6, 1200)
    fig, ax = plt.subplots()
    for k in range(60):
        fk = randnfun_np(0.05, dom6, 100 + k)
        solk = solve_ivp(
            lambda t, y: [y[0] - y[0] ** 3 / 4 + 1.2 * fk(t)], dom6,
            [0.0], t_eval=ts6, max_step=0.005, rtol=1e-6)
        ax.plot(solk.t, solk.y[0], linewidth=0.5)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title("lambda = 0.05, 60 paths", fontsize=10)
    save(fig, "PhaseLocking_03.png")


def pitchfork():
    """ode-random/Pitchfork — noisy pitchfork bifurcation sweep."""
    dom = (0.0, 100.0)
    ts = np.linspace(*dom, 8000)
    f = randnfun_np(0.3, dom, 2)

    # a(t) sweeps from -1 to 1: y' = a(t) y - y^3 + noise
    for j, eps in enumerate((0.03, 0.2), 1):
        sol = solve_ivp(
            lambda t, y: [(-1 + 2 * t / 100) * y[0] - y[0] ** 3
                          + eps * f(t)],
            dom, [0.0], t_eval=ts, max_step=0.02, rtol=1e-7)
        fig, ax = plt.subplots()
        ax.plot(sol.t, sol.y[0], color=CHEBFUN_BLUE, linewidth=0.7)
        aa = -1 + 2 * ts / 100
        ax.plot(ts, np.sqrt(np.maximum(aa, 0)), "r--", linewidth=0.7)
        ax.plot(ts, -np.sqrt(np.maximum(aa, 0)), "r--", linewidth=0.7)
        ax.grid(True, alpha=0.4, linewidth=0.4)
        ax.set_title(f"noisy pitchfork, noise {eps:g}", fontsize=10)
        ax.set_xlabel("t")
        save(fig, f"Pitchfork_{j:02d}.png")


def random2sde():
    """ode-random/Random2SDE — smooth-noise ODE vs SDE limit."""
    dom = (0.0, 10.0)
    ts = np.linspace(*dom, 3000)
    fig, ax = plt.subplots()
    for lam, color in ((1.0, CHEBFUN_BLUE), (0.2, ORANGE),
                       (0.05, "g")):
        f = randnfun_np(lam, dom, 7)
        sol = solve_ivp(lambda t, y: [-y[0] + f(t)], dom, [0.0],
                        t_eval=ts, max_step=0.01, rtol=1e-7)
        ax.plot(sol.t, sol.y[0], color=color, linewidth=0.7,
                label=f"lambda = {lam:g}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_xlabel("t")
    ax.set_title("smooth random ODE approaching the SDE limit",
                 fontsize=9)
    save(fig, "Random2SDE_01.png")


def randomonasphere():
    """ode-random/RandomOnASphere — random walk on the sphere."""
    dom = (0.0, 50.0)
    ts = np.linspace(*dom, 8000)
    fx = randnfun_np(0.5, dom, 11)
    fy = randnfun_np(0.5, dom, 12)
    fz = randnfun_np(0.5, dom, 13)

    def rhs(t, y):
        v = np.array([fx(t), fy(t), fz(t)])
        # project onto the tangent plane to stay on the sphere
        y = np.asarray(y)
        v = v - (v @ y) * y
        return v

    sol = solve_ivp(rhs, dom, [0.0, 0.0, 1.0], t_eval=ts,
                    max_step=0.02, rtol=1e-8)
    P = sol.y / np.linalg.norm(sol.y, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    th = np.linspace(0, PI, 40)
    ph = np.linspace(0, 2 * PI, 80)
    TH, PH = np.meshgrid(th, ph)
    ax.plot_surface(np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH),
                    np.cos(TH), color=(0.9, 0.9, 0.9), alpha=0.35,
                    linewidth=0)
    ax.plot3D(P[0], P[1], P[2], color=CHEBFUN_BLUE, linewidth=0.6)
    ax.set_box_aspect((1, 1, 1))
    ax.axis("off")
    save(fig, "RandomOnASphere_01.png")

    fig, ax = plt.subplots()
    ax.plot(ts, P[2], color=CHEBFUN_BLUE, linewidth=0.7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_xlabel("t")
    ax.set_title("z-coordinate of the random path", fontsize=10)
    save(fig, "RandomOnASphere_02.png")


def randomswitching():
    """ode-random/RandomSwitching — switching between two systems."""
    dom = (0.0, 60.0)
    ts = np.linspace(*dom, 6000)
    rng = np.random.default_rng(5)
    # random telegraph switching times
    switch_times = np.cumsum(rng.exponential(3.0, 60))
    switch_times = switch_times[switch_times < 60]

    def state(t):
        return int(np.searchsorted(switch_times, t) % 2)

    fig, ax = plt.subplots()
    sig = np.array([state(t) for t in ts])
    ax.step(ts, sig, color=CHEBFUN_BLUE, linewidth=0.8)
    ax.set_ylim(-0.2, 1.2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("random switching signal", fontsize=10)
    save(fig, "RandomSwitching_01.png")

    # switch between two stable spirals with different centers
    A1 = np.array([[-0.1, -1.0], [1.0, -0.1]])
    A2 = np.array([[-0.1, -2.0], [2.0, -0.1]])
    c1 = np.array([1.0, 0.0])
    c2 = np.array([-1.0, 0.0])

    def rhs(t, y):
        if state(t) == 0:
            return A1 @ (np.asarray(y) - c1)
        return A2 @ (np.asarray(y) - c2)

    sol = solve_ivp(rhs, dom, [2.0, 0.0], t_eval=ts, max_step=0.01,
                    rtol=1e-8)
    fig, ax = plt.subplots()
    ax.plot(sol.y[0], sol.y[1], color=CHEBFUN_BLUE, linewidth=0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("trajectory under random switching", fontsize=10)
    save(fig, "RandomSwitching_02.png")

    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[0], color=CHEBFUN_BLUE, linewidth=0.7)
    ax.plot(sol.t, sol.y[1], color=ORANGE, linewidth=0.7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_xlabel("t")
    save(fig, "RandomSwitching_03.png")

    # decay-to-consensus panels: contracting switched linear system
    A1c = np.array([[-0.4, -1.0], [1.0, -0.4]])
    A2c = np.array([[-0.4, -2.0], [2.0, -0.4]])

    def rhs_c(t, y):
        A = A1c if state(t) == 0 else A2c
        return A @ np.asarray(y)

    ts40 = np.linspace(0, 40, 4000)
    solc = solve_ivp(rhs_c, (0, 40), [2.0, -1.0], t_eval=ts40,
                     max_step=0.01, rtol=1e-9)
    nrm = np.linalg.norm(solc.y, axis=0)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(solc.t, solc.y[0], color=CHEBFUN_BLUE, linewidth=0.7)
    ax1.plot(solc.t, solc.y[1], color=ORANGE, linewidth=0.7)
    ax1.set_title("u and v on linear scale", fontsize=8)
    ax2.semilogy(solc.t, np.maximum(nrm, 1e-8), "k", linewidth=0.8)
    ax2.set_title("norm of (u,v) on log scale", fontsize=8)
    for a in (ax1, ax2):
        a.grid(True, alpha=0.4, linewidth=0.4)
        a.tick_params(labelsize=6)
    save(fig, "RandomSwitching_04.png")


def tunnelling():
    """ode-random/Tunnelling — double-well hopping statistics."""
    dom = (0.0, 200.0)
    ts = np.linspace(*dom, 12000)

    for j, eps in enumerate((0.5, 0.8), 1):
        f = randnfun_np(0.5, dom, 20 + j)
        sol = solve_ivp(
            lambda t, y: [y[0] - y[0] ** 3 + eps * f(t)], dom, [1.0],
            t_eval=ts, max_step=0.05, rtol=1e-7)
        fig, ax = plt.subplots()
        ax.plot(sol.t, sol.y[0], color=CHEBFUN_BLUE, linewidth=0.5)
        ax.axhline(1, color="r", linewidth=0.5, linestyle="--")
        ax.axhline(-1, color="r", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.4, linewidth=0.4)
        ax.set_xlabel("t")
        ax.set_title(f"tunnelling, noise {eps:g}", fontsize=10)
        save(fig, f"Tunnelling_{j:02d}.png")

    # long high-noise run: dense hopping to t = 800
    dom8 = (0.0, 800.0)
    ts8 = np.linspace(*dom8, 20000)
    f8 = randnfun_np(0.5, dom8, 30)
    sol8 = solve_ivp(
        lambda t, y: [y[0] - y[0] ** 3 + 1.1 * f8(t)], dom8, [1.0],
        t_eval=ts8, max_step=0.05, rtol=1e-6)
    fig, ax = plt.subplots()
    ax.plot(sol8.t, sol8.y[0], color=CHEBFUN_BLUE, linewidth=0.3)
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title("Larger noise means faster tunnelling", fontsize=10)
    save(fig, "Tunnelling_03.png")


def whitenoiseparadox():
    """ode-random/WhiteNoiseParadox — energy grows as lambda -> 0."""
    dom = (-1.0, 1.0)
    ts = np.linspace(*dom, 4000)
    fig, axes = plt.subplots(1, 3)
    for a, lam, lbl in zip(axes, (0.25, 1 / 16, 1 / 64),
                           ("1/4", "1/16", "1/64")):
        f = randnfun_np(lam, dom, 9)
        a.plot(ts, f(ts), color=CHEBFUN_BLUE, linewidth=0.5)
        a.set_ylim(-30, 30)
        a.set_xlim(-1, 1)
        a.set_title(f"lambda = {lbl}", fontsize=8)
        a.tick_params(labelsize=6)
    save(fig, "WhiteNoiseParadox_01.png")


PAGES = {
    "Consensus": consensus,
    "GBM": gbm,
    "LevelHopping": levelhopping,
    "PhaseLocking": phaselocking,
    "Pitchfork": pitchfork,
    "Random2SDE": random2sde,
    "RandomOnASphere": randomonasphere,
    "RandomSwitching": randomswitching,
    "Tunnelling": tunnelling,
    "WhiteNoiseParadox": whitenoiseparadox,
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
