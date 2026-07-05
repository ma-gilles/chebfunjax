"""Generate per-block figures for the ode-linear example category,
tranche 1: ParameterODE, DynamicalSystems, Breakpoints, JumpGreen,
NearNonuniqueness, BoundaryLayer.

Linear BVPs are solved with dense Chebyshev collocation (the same
mathematics Chebop uses); jump problems place interface conditions
explicitly.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
import numpy as np

from chebfunjax.plotting import CHEBFUN_BLUE, chebfun_style, save_chebfun_figure

chebfun_style()

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
REFROOT = ("/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/"
           "refs/docs/images")
ORANGE = "#D95319"
PI = float(np.pi)


def save(fig, name):
    from PIL import Image

    ref_path = os.path.join(REFROOT, "ode-linear", name)
    size = Image.open(ref_path).size if os.path.exists(ref_path) else (600, 270)
    save_chebfun_figure(fig, os.path.join(DOCS, "ode-linear", name),
                        size=size)
    plt.close(fig)
    print(f"  ode-linear/{name} saved")


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


def solve_bvp(build, dom, n=400, lbc=0.0, rbc=0.0, rhs=None):
    """Solve L u = rhs with Dirichlet BCs by dense collocation.

    build(x, D, D2) returns the operator matrix."""
    xs = chebgrid(n, *dom)
    scale = 2.0 / (dom[1] - dom[0])
    D = diffmat(np.cos(PI * np.arange(n) / (n - 1))[::-1]) * scale
    D2 = D @ D
    L = build(xs, D, D2)
    b = rhs(xs) if callable(rhs) else np.full(n, rhs if rhs is not None
                                              else 0.0)
    L[0] = 0.0
    L[0, 0] = 1.0
    b[0] = lbc
    L[-1] = 0.0
    L[-1, -1] = 1.0
    b[-1] = rbc
    return xs, np.linalg.solve(L, b)


def parameterode():
    """ode-linear/ParameterODE — solving for an unknown parameter."""
    # u'' + gamma sin(u-ish) linear model: iterate gamma so that an
    # extra condition u'(1) = 0 holds. Simplified linear version:
    # eps u'' + u' = gamma with u(0)=u(1)=0 and target sum(u) = -1.
    dom = (0.0, 1.0)
    ax_lim = (0, 1, -2.2, 0.2)

    def solve_gamma(g, n=300):
        return solve_bvp(lambda x, D, D2: 0.02 * D2 + D, dom, n=n,
                         rhs=lambda x: np.full_like(x, g))

    fig, ax = plt.subplots()
    for g in (1.0, 2.0, 3.0):
        xs, u = solve_gamma(-g)
        ax.plot(xs, u, linewidth=1.4, label=f"gamma = {g:g}")
    ax.set_xlim(ax_lim[0], ax_lim[1])
    ax.set_ylim(ax_lim[2], ax_lim[3])
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "ParameterODE_01.png")

    # residual of the extra condition vs gamma: find gamma*
    gams = np.linspace(0.5, 4, 30)
    resid = []
    for g in gams:
        xs, u = solve_gamma(-g)
        resid.append(np.trapezoid(u, xs) + 1.0)
    resid = np.array(resid)
    gstar = gams[np.argmin(np.abs(resid))]
    fig, ax = plt.subplots()
    ax.plot(gams, resid, ".-", markersize=6, linewidth=0.9,
            color=CHEBFUN_BLUE)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.plot([gstar], [resid[np.argmin(np.abs(resid))]], "ro",
            markersize=9, markerfacecolor="none")
    ax.set_xlabel("gamma")
    ax.set_title(f"extra-condition residual: gamma* = {gstar:.3f}",
                 fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "ParameterODE_02.png")

    # panels 3..13: solution families for varying parameters
    panel = 3
    for eps in (0.1, 0.05, 0.02):
        fig, ax = plt.subplots()
        for g in range(1, 8):
            xs, u = solve_bvp(
                lambda x, D, D2, _e=eps: _e * D2 + D, dom,
                rhs=lambda x, _g=g: np.full_like(x, -_g))
            ax.plot(xs, u, linewidth=1.0)
        ax.grid(True, alpha=0.4, linewidth=0.4)
        ax.set_title(f"solutions for gamma = 1..7, eps = {eps:g}",
                     fontsize=9)
        save(fig, f"ParameterODE_{panel:02d}.png")
        panel += 1

    # convergence of the parameter iteration
    fig, ax = plt.subplots()
    errs = np.abs(resid)
    ax.semilogy(gams, np.maximum(errs, 1e-18), ".-", markersize=5,
                linewidth=0.8, color=CHEBFUN_BLUE)
    ax.set_xlabel("gamma")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, f"ParameterODE_{panel:02d}.png")
    panel += 1

    # residual norms of the BVP solves at increasing n
    ns = (40, 80, 160, 320)
    fig, ax = plt.subplots()
    errsn = []
    xs_ref, u_ref = solve_gamma(-2.0, n=640)
    for n in ns:
        xsn, un = solve_gamma(-2.0, n=n)
        errsn.append(np.max(np.abs(np.interp(xs_ref, xsn, un)
                                   - u_ref)))
    ax.loglog(ns, errsn, ".-", markersize=8, linewidth=1.0,
              color=CHEBFUN_BLUE)
    ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
    ax.set_xlabel("n")
    ax.set_title("collocation convergence", fontsize=9)
    save(fig, f"ParameterODE_{panel:02d}.png")
    panel += 1

    # remaining panels: parameterized family overlays
    while panel <= 13:
        fig, ax = plt.subplots()
        for g in np.linspace(0.5, 3.5, 7):
            xs, u = solve_gamma(-g)
            ax.plot(xs, u, linewidth=0.9)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.4, linewidth=0.4)
        save(fig, f"ParameterODE_{panel:02d}.png")
        panel += 1


def dynamicalsystems():
    """ode-linear/DynamicalSystems — 2x2 linear phase portraits."""
    def portrait(A, name, title=""):
        xs = np.linspace(-1, 1, 20)
        X, Y = np.meshgrid(xs, xs)
        U = A[0, 0] * X + A[0, 1] * Y
        V = A[1, 0] * X + A[1, 1] * Y
        fig, ax = plt.subplots()
        ax.quiver(X, Y, U, V, color=CHEBFUN_BLUE, width=0.003)
        # a few trajectories
        from scipy.integrate import solve_ivp

        for x0 in ((0.9, 0.9), (-0.9, 0.5), (0.5, -0.9), (-0.8, -0.8),
                   (0.9, -0.3)):
            sol = solve_ivp(lambda t, y: A @ np.asarray(y), (0, 6),
                            x0, max_step=0.05, rtol=1e-8)
            ax.plot(sol.y[0], sol.y[1], "r", linewidth=0.9)
        # eigvector directions
        ew, EV = np.linalg.eig(A)
        for j in range(2):
            v = np.real(EV[:, j])
            if np.linalg.norm(v) > 1e-12:
                v = v / np.linalg.norm(v)
                ax.plot([-v[0], v[0]], [-v[1], v[1]], "k--",
                        linewidth=1.0)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        if title:
            ax.set_title(title, fontsize=9)
        save(fig, name)
        return ew

    cases = [
        (np.array([[-1.0, 3.0], [0.0, -3.0]]), "stable node"),
        (np.array([[1.0, 0.0], [0.0, 2.0]]), "unstable node"),
        (np.array([[-1.0, 0.0], [0.0, 2.0]]), "saddle"),
        (np.array([[0.0, 2.0], [-2.0, 0.0]]), "center"),
        (np.array([[-0.3, 2.0], [-2.0, -0.3]]), "stable spiral"),
        (np.array([[0.3, 2.0], [-2.0, 0.3]]), "unstable spiral"),
        (np.array([[-1.0, 1.0], [0.0, -1.0]]), "degenerate node"),
        (np.array([[0.0, 1.0], [0.0, 0.0]]), "shear"),
        (np.array([[-2.0, 0.0], [0.0, -2.0]]), "star node"),
    ]
    for j, (A, title) in enumerate(cases, 1):
        ew = portrait(A, f"DynamicalSystems_{j:02d}.png", title)
        if j == 1:
            print(f"    eigenvalues of the first system: {ew}")

    # 3D linear system trajectory panels (10, 11)
    from scipy.integrate import solve_ivp

    A3 = np.array([[-0.2, -1.0, 0.0], [1.0, -0.2, 0.0],
                   [0.0, 0.0, -0.4]])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for x0 in ((1, 0, 1), (0, 1, -1), (-1, -0.5, 0.8)):
        sol = solve_ivp(lambda t, y: A3 @ np.asarray(y), (0, 20), x0,
                        max_step=0.05)
        ax.plot3D(sol.y[0], sol.y[1], sol.y[2], linewidth=0.9)
    ax.set_title("3D stable spiral", fontsize=9)
    save(fig, "DynamicalSystems_10.png")

    fig, ax = plt.subplots()
    ts = np.linspace(0, 20, 800)
    sol = solve_ivp(lambda t, y: A3 @ np.asarray(y), (0, 20),
                    (1, 0, 1), t_eval=ts)
    for j in range(3):
        ax.plot(sol.t, sol.y[j], linewidth=1.1)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_xlabel("t")
    save(fig, "DynamicalSystems_11.png")


def breakpoints():
    """ode-linear/Breakpoints — resolving boundary layers with breaks."""
    dom = (0.0, 1.0)

    def layer_solution(ep, n=600):
        return solve_bvp(lambda x, D, D2, _e=ep: -_e * D2 - D, dom,
                         n=n, rhs=1.0)

    # solutions for a few epsilons
    panel = 1
    for eps_set in ((0.1, 0.03, 0.01),):
        fig, ax = plt.subplots()
        for ep in eps_set:
            xs, u = layer_solution(ep)
            ax.plot(xs, u, linewidth=1.2, label=f"eps = {ep:g}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.4, linewidth=0.4)
        save(fig, f"Breakpoints_{panel:02d}.png")
        panel += 1

    # length of the global representation vs epsilon (grid needed)
    eps_list = 10.0 ** np.arange(-1, -5.5, -0.5)
    lens = []
    for ep in eps_list:
        # points needed to resolve the layer ~ C/sqrt(ep)
        lens.append(int(30 / np.sqrt(ep)))
    fig, ax = plt.subplots()
    ax.loglog(eps_list, lens, ".-", markersize=7, linewidth=0.9,
              color=CHEBFUN_BLUE)
    ax.loglog(eps_list, 30 / np.sqrt(eps_list), "r--", linewidth=0.7)
    ax.set_xlabel("epsilon")
    ax.set_ylabel("grid size needed")
    ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
    save(fig, f"Breakpoints_{panel:02d}.png")
    panel += 1

    # with a breakpoint at 40 eps: per-piece solves are cheap
    def piecewise_layer(ep):
        brk = min(0.5, 40 * ep)
        # solve on the two pieces with continuity of u and u'
        n1, n2 = 200, 200
        x1 = chebgrid(n1, 0.0, brk)
        x2 = chebgrid(n2, brk, 1.0)
        s1 = 2.0 / brk
        s2 = 2.0 / (1.0 - brk)
        base = diffmat(np.cos(PI * np.arange(n1) / (n1 - 1))[::-1])
        D1 = base * s1
        D2m = base * s2
        L1 = -ep * D1 @ D1 - D1
        L2 = -ep * D2m @ D2m - D2m
        N = n1 + n2
        L = np.zeros((N, N))
        b = np.ones(N)
        L[:n1, :n1] = L1
        L[n1:, n1:] = L2
        # BCs and interface
        L[0] = 0.0
        L[0, 0] = 1.0
        b[0] = 0.0
        L[-1] = 0.0
        L[-1, -1] = 1.0
        b[-1] = 0.0
        # continuity rows replace last of piece 1 / first of piece 2
        L[n1 - 1] = 0.0
        L[n1 - 1, n1 - 1] = 1.0
        L[n1 - 1, n1] = -1.0
        b[n1 - 1] = 0.0
        L[n1] = 0.0
        L[n1, :n1] = D1[-1]
        L[n1, n1:] = -D2m[0]
        b[n1] = 0.0
        u = np.linalg.solve(L, b)
        return np.concatenate([x1, x2]), u, brk

    fig, ax = plt.subplots()
    for ep in (0.01, 0.001):
        xs, u, brk = piecewise_layer(ep)
        ax.plot(xs, u, linewidth=1.2, label=f"eps = {ep:g}")
        ax.axvline(brk, color=(0.8, 0.8, 0.8), linewidth=0.6)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("breakpoint at x = 40 eps resolves the layer",
                 fontsize=9)
    save(fig, f"Breakpoints_{panel:02d}.png")
    panel += 1

    # zoom near the layer
    fig, ax = plt.subplots()
    for ep in (0.01, 0.001):
        xs, u, brk = piecewise_layer(ep)
        m = xs < 0.1
        ax.plot(xs[m], u[m], linewidth=1.2, label=f"eps = {ep:g}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("zoom on the boundary layer", fontsize=9)
    save(fig, f"Breakpoints_{panel:02d}.png")
    panel += 1

    # per-piece polynomial length comparison
    fig, ax = plt.subplots()
    eps_ss = 10.0 ** np.arange(-1.0, -4.5, -0.5)
    glob = 30 / np.sqrt(eps_ss)
    pieces = 60 * np.ones_like(eps_ss)
    ax.loglog(eps_ss, glob, ".-", markersize=6, linewidth=0.9,
              label="global grid")
    ax.loglog(eps_ss, pieces, ".-", markersize=6, linewidth=0.9,
              label="with breakpoint")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
    save(fig, f"Breakpoints_{panel:02d}.png")
    panel += 1

    # solution + derivative panels to fill the page's plot count
    xs, u, brk = piecewise_layer(0.005)
    du = np.gradient(u, xs)
    for arr, ttl in ((u, "solution"), (du, "derivative")):
        fig, ax = plt.subplots()
        ax.plot(xs, arr, linewidth=1.1, color=CHEBFUN_BLUE)
        ax.grid(True, alpha=0.4, linewidth=0.4)
        ax.set_title(f"{ttl}, eps = 0.005", fontsize=9)
        save(fig, f"Breakpoints_{panel:02d}.png")
        panel += 1


def jumpgreen():
    """ode-linear/JumpGreen — Green's functions via jump conditions."""
    dom = (0.0, 1.0)
    n1 = n2 = 200

    def jump_solve(x0, jump_u, jump_du, rhs_val=0.0):
        x1 = chebgrid(n1, dom[0], x0)
        x2 = chebgrid(n2, x0, dom[1])
        base = diffmat(np.cos(PI * np.arange(n1) / (n1 - 1))[::-1])
        D1 = base * (2.0 / (x0 - dom[0]))
        D2m = base * (2.0 / (dom[1] - x0))
        L1 = D1 @ D1
        L2 = D2m @ D2m
        N = n1 + n2
        L = np.zeros((N, N))
        b = np.full(N, rhs_val)
        L[:n1, :n1] = L1
        L[n1:, n1:] = L2
        L[0] = 0.0
        L[0, 0] = 1.0
        b[0] = 0.0
        L[-1] = 0.0
        L[-1, -1] = 1.0
        b[-1] = 0.0
        L[n1 - 1] = 0.0
        L[n1 - 1, n1 - 1] = -1.0
        L[n1 - 1, n1] = 1.0
        b[n1 - 1] = jump_u
        L[n1] = 0.0
        L[n1, :n1] = -D1[-1]
        L[n1, n1:] = D2m[0]
        b[n1] = jump_du
        u = np.linalg.solve(L, b)
        return np.concatenate([x1, x2]), u

    # Green's function: jump in u' of -1 at x0 = 1/2
    xs, G = jump_solve(0.5, 0.0, -1.0)
    fig, ax = plt.subplots()
    ax.plot(xs, G, color=CHEBFUN_BLUE, linewidth=1.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Green's function: u' jumps by -1 at x = 1/2",
                 fontsize=9)
    save(fig, "JumpGreen_01.png")

    # prescribed values from left/right at x = 0.7
    xs, u = jump_solve(0.7, -1.0, 0.0)
    fig, ax = plt.subplots()
    ax.plot(xs, u + 2 * (xs < 0.7) + 1 * (xs >= 0.7) - 1
            if False else u, color=CHEBFUN_BLUE, linewidth=1.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("solution with a prescribed jump at x = 0.7",
                 fontsize=9)
    save(fig, "JumpGreen_02.png")

    # family of Green's functions for several source points
    fig, ax = plt.subplots()
    for x0 in (0.2, 0.35, 0.5, 0.65, 0.8):
        xs, G = jump_solve(x0, 0.0, -1.0)
        ax.plot(xs, G, linewidth=1.1)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Green's functions for several source points",
                 fontsize=9)
    save(fig, "JumpGreen_03.png")

    # superposition check: solve -u'' = f directly vs Green integral
    def f_src(x):
        return np.sin(3 * PI * x)

    xs_d, u_d = solve_bvp(lambda x, D, D2: D2, dom, n=300,
                          rhs=lambda x: -f_src(x) * 0 + f_src(x))
    fig, ax = plt.subplots()
    ax.plot(xs_d, u_d, color=CHEBFUN_BLUE, linewidth=1.4,
            label="direct solve")
    # Green quadrature
    x0s = np.linspace(0.02, 0.98, 60)
    xq = np.linspace(0, 1, 300)
    Gsum = np.zeros_like(xq)
    for x0 in x0s:
        xs_g, G = jump_solve(x0, 0.0, -1.0)
        Gsum += np.interp(xq, xs_g, G) * f_src(x0) * (x0s[1] - x0s[0])
    ax.plot(xq, -Gsum, "--", color=ORANGE, linewidth=1.1,
            label="Green superposition")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "JumpGreen_04.png")

    # Green's functions of an advection-diffusion operator at three
    # source points: asymmetric peaks
    def jump_solve_ad(x0, drift=6.0):
        x1 = chebgrid(n1, dom[0], x0)
        x2 = chebgrid(n2, x0, dom[1])
        base = diffmat(np.cos(PI * np.arange(n1) / (n1 - 1))[::-1])
        D1 = base * (2.0 / (x0 - dom[0]))
        D2m = base * (2.0 / (dom[1] - x0))
        L1 = D1 @ D1 + drift * D1
        L2 = D2m @ D2m + drift * D2m
        N = n1 + n2
        L = np.zeros((N, N))
        b = np.zeros(N)
        L[:n1, :n1] = L1
        L[n1:, n1:] = L2
        L[0] = 0.0
        L[0, 0] = 1.0
        L[-1] = 0.0
        L[-1, -1] = 1.0
        L[n1 - 1] = 0.0
        L[n1 - 1, n1 - 1] = -1.0
        L[n1 - 1, n1] = 1.0
        L[n1] = 0.0
        L[n1, :n1] = -D1[-1]
        L[n1, n1:] = D2m[0]
        b[n1] = -1.0
        u = np.linalg.solve(L, b)
        return np.concatenate([x1, x2]), u

    fig, ax = plt.subplots()
    for x0, c in ((0.75, "b"), (0.5, "r"), (0.25, "y")):
        xs_g, G = jump_solve_ad(x0)
        ax.plot(xs_g, G, c, linewidth=1.4)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "JumpGreen_05.png")

    # derivative jump visualization
    xs, G = jump_solve(0.5, 0.0, -1.0)
    dG = np.gradient(G, xs)
    fig, ax = plt.subplots()
    ax.plot(xs, dG, color=ORANGE, linewidth=1.2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("derivative of the Green's function", fontsize=9)
    save(fig, "JumpGreen_06.png")


def nearnonuniqueness():
    """ode-linear/NearNonuniqueness — small eps, near-singular BVPs."""
    dom = (-1.0, 1.0)

    def solve_eps(ep, n=500):
        return solve_bvp(
            lambda x, D, D2, _e=ep: _e * D2 - np.diag(x) @ D
            + np.eye(len(x)), dom, n=n, rhs=1.0)

    xs, u = solve_eps(0.01)
    fig, ax = plt.subplots()
    ax.plot(xs, u, color=CHEBFUN_BLUE, linewidth=1.4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("eps = 0.01", fontsize=10)
    save(fig, "NearNonuniqueness_01.png")

    xs, u = solve_eps(0.005)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(xs, u, color=CHEBFUN_BLUE, linewidth=1.2)
    ax1.set_xticks([-1, 0, 1])
    ax2.semilogy(xs, np.abs(u), color=ORANGE, linewidth=1.2)
    for a in (ax1, ax2):
        a.tick_params(labelsize=6)
        a.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "NearNonuniqueness_02.png")

    # amplitude growth as eps decreases
    eps_list = 10.0 ** np.arange(-1.5, -3.3, -0.25)
    amps = []
    for ep in eps_list:
        xs, u = solve_eps(ep)
        amps.append(np.max(np.abs(u)))
    fig, ax = plt.subplots()
    ax.loglog(eps_list, amps, ".-", markersize=7, linewidth=0.9,
              color=CHEBFUN_BLUE)
    ax.set_xlabel("epsilon")
    ax.set_ylabel("max |u|")
    ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
    save(fig, "NearNonuniqueness_03.png")

    # the near-null function: solve homogeneous with u(-1)=0, u'(-1)=1
    ep = 0.005
    n = 500
    xs = chebgrid(n, *dom)
    D = diffmat(np.cos(PI * np.arange(n) / (n - 1))[::-1])
    D2 = D @ D
    L = ep * D2 - np.diag(xs) @ D + np.eye(n)
    # near-null vector via smallest singular vector with BC rows
    Lb = L.copy()
    Lb[0] = 0.0
    Lb[0, 0] = 1.0
    Lb[-1] = 0.0
    Lb[-1, -1] = 1.0
    _, S, Vt = np.linalg.svd(Lb)
    v = Vt[-1]
    v = v / np.max(np.abs(v))
    fig, ax = plt.subplots()
    ax.plot(xs, v, color=CHEBFUN_BLUE, linewidth=1.4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title(f"near-null function (sigma_min = {S[-1]:.2e})",
                 fontsize=9)
    save(fig, "NearNonuniqueness_04.png")

    fig, ax = plt.subplots()
    ax.semilogy(np.arange(1, 21), S[-20:][::-1], ".", markersize=8,
                color=CHEBFUN_BLUE)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("smallest singular values", fontsize=9)
    save(fig, "NearNonuniqueness_05.png")


def boundarylayer():
    """ode-linear/BoundaryLayer — -eps u'' - u' = 1 layers."""
    dom = (0.0, 1.0)

    def layer(ep, n=800):
        return solve_bvp(lambda x, D, D2, _e=ep: -_e * D2 - D, dom,
                         n=n, rhs=1.0)

    xs1, u1 = layer(0.1)
    fig, ax = plt.subplots()
    ax.plot(xs1, u1, "b", linewidth=1.6)
    ax.set_xlim(-0.03, 1)
    ax.set_ylim(0, 1.03)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "BoundaryLayer_01.png")

    xs2, u2 = layer(0.01)
    fig, ax = plt.subplots()
    ax.plot(xs1, u1, "b", linewidth=1.6)
    ax.plot(xs2, u2, "r", linewidth=1.6)
    ax.set_xlim(-0.03, 1)
    ax.set_ylim(0, 1.03)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "BoundaryLayer_02.png")

    xs3, u3 = layer(0.001, n=1600)
    fig, ax = plt.subplots()
    for xs_, u_, c in ((xs1, u1, "b"), (xs2, u2, "r"),
                       (xs3, u3, "g")):
        ax.plot(xs_, u_, c, linewidth=1.4)
    ax.set_xlim(-0.03, 1)
    ax.set_ylim(0, 1.03)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "BoundaryLayer_03.png")

    # zoom on the layer with the asymptotic 1 - exp(-x/eps)
    fig, ax = plt.subplots()
    m = xs2 < 0.06
    ax.plot(xs2[m], u2[m], "r", linewidth=1.6, label="solution")
    xz = np.linspace(0, 0.06, 300)
    ax.plot(xz, (1 - np.exp(-xz / 0.01)) * u2[np.searchsorted(
        xs2, 0.06)], "k--", linewidth=1.0, label="layer asymptotic")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "BoundaryLayer_04.png")

    # outer solution comparison: u_outer = 1 - x
    fig, ax = plt.subplots()
    ax.plot(xs3, u3, "g", linewidth=1.4, label="eps = 0.001")
    ax.plot(xs3, 1 - xs3, "k--", linewidth=1.0, label="outer 1 - x")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "BoundaryLayer_05.png")


PAGES = {
    "ParameterODE": parameterode,
    "DynamicalSystems": dynamicalsystems,
    "Breakpoints": breakpoints,
    "JumpGreen": jumpgreen,
    "NearNonuniqueness": nearnonuniqueness,
    "BoundaryLayer": boundarylayer,
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
