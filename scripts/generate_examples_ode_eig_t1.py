"""Generate per-block figures for the ode-eig example category,
tranche 1: Eigenstates, NullSpace, Randfuneig, SolarQDA, Landscape,
Drum, DoubleWell.
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

    ref_path = os.path.join(REFROOT, "ode-eig", name)
    size = Image.open(ref_path).size if os.path.exists(ref_path) else (600, 270)
    save_chebfun_figure(fig, os.path.join(DOCS, "ode-eig", name),
                        size=size)
    plt.close(fig)
    print(f"  ode-eig/{name} saved")


def _schrodinger(V_np, dom, n=1200, h=0.1, k=10):
    """Lowest k eigenpairs of -h^2 u'' + V u on dom, Dirichlet BCs.

    Dense second-order FD discretization; returns (xs, evals, evecs)
    with evecs L2-normalized columns."""
    a, b = dom
    xs = np.linspace(a, b, n + 2)[1:-1]
    dx = xs[1] - xs[0]
    main = 2 * np.ones(n) * h**2 / dx**2 + V_np(xs)
    off = -np.ones(n - 1) * h**2 / dx**2
    import scipy.linalg as sla

    evals, evecs = sla.eigh_tridiagonal(main, off,
                                        select="i",
                                        select_range=(0, k - 1))
    evecs = evecs / np.sqrt(dx)
    return xs, evals, evecs


def _quantumstates_plot(V_np, dom=(-3.0, 3.0), k=10, h=0.1, name=None,
                        scale=None):
    xs, evals, evecs = _schrodinger(V_np, dom, h=h, k=k)
    fig, ax = plt.subplots()
    xv = np.linspace(dom[0], dom[1], 1500)
    ax.plot(xv, V_np(xv), "k", linewidth=1.4)
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    sc = scale if scale else 0.35 * np.median(np.diff(evals))
    for j in range(k):
        ax.plot(xs, evals[j] + sc * evecs[:, j] / np.max(
            np.abs(evecs[:, j])), color=cyc[j % len(cyc)],
            linewidth=0.8)
        ax.axhline(evals[j], color=(0.85, 0.85, 0.85), linewidth=0.4,
                   zorder=0)
    ax.set_xlim(*dom)
    if name:
        save(fig, name)
    return evals


def eigenstates():
    """ode-eig/Eigenstates — quantumstates() for various potentials."""
    x2 = lambda x: np.asarray(x) ** 2
    _quantumstates_plot(x2, name="Eigenstates_01.png")
    _quantumstates_plot(x2, k=60, name="Eigenstates_02.png")
    _quantumstates_plot(x2, h=0.01, k=10, name="Eigenstates_03.png")
    _quantumstates_plot(x2, h=0.5, k=12, name="Eigenstates_04.png")

    well10 = lambda x: 10.0 - 10.0 * (np.abs(np.asarray(x)) < 1)
    _quantumstates_plot(well10, k=10, name="Eigenstates_05.png")

    well1 = lambda x: 1.0 - (np.abs(np.asarray(x)) < 1).astype(float)
    _quantumstates_plot(well1, k=14, name="Eigenstates_06.png")

    _quantumstates_plot(lambda x: np.abs(np.asarray(x)),
                        name="Eigenstates_07.png")
    _quantumstates_plot(lambda x: np.sqrt(np.abs(np.asarray(x)) + 0.1),
                        name="Eigenstates_08.png")

    step = lambda x: 0.5 * ((np.abs(np.asarray(x) - 0.5) < 0.5)
                            .astype(float))
    _quantumstates_plot(step, k=12, name="Eigenstates_09.png")

    gauss = lambda x: 0.5 * np.exp(-2 * (np.asarray(x) - 0.5) ** 2)
    _quantumstates_plot(gauss, k=12, name="Eigenstates_10.png")


def nullspace():
    """ode-eig/NullSpace — null vectors of differential operators."""
    # L = d^2/dx^2 on [-1, 1] with NO boundary conditions: null dim 2
    n = 300
    xs = np.cos(PI * np.arange(n) / (n - 1))[::-1]

    def diffmat(x):
        # barycentric differentiation matrix (Chebyshev points)
        N = len(x)
        c = np.ones(N)
        c[0] = c[-1] = 2.0
        c *= (-1.0) ** np.arange(N)
        X = x[:, None] - x[None, :]
        D = (c[:, None] / c[None, :]) / (X + np.eye(N))
        D -= np.diag(D.sum(axis=1))
        return D

    D = diffmat(xs)
    L = D @ D
    U, S, Vt = np.linalg.svd(L)
    V = Vt[-2:].T
    # orthonormalize in L2 and rotate for a clean picture
    Q, _ = np.linalg.qr(V)

    fig, ax = plt.subplots()
    ax.plot(xs, Q[:, 0], linewidth=1.6)
    ax.plot(xs, Q[:, 1], linewidth=1.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("null space of u'' (no BCs): dimension 2",
                 fontsize=10)
    save(fig, "NullSpace_01.png")

    # with one BC u(-1) = 0: null dim 1
    L1 = L.copy()
    L1[0] = 0.0
    L1[0, 0] = 1.0
    U, S, Vt = np.linalg.svd(L1)
    v = Vt[-1]
    v = v / np.max(np.abs(v)) * np.sign(v[np.argmax(np.abs(v))])
    fig, ax = plt.subplots()
    ax.plot(xs, v, linewidth=1.6, color=CHEBFUN_BLUE)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("null space with u(-1) = 0", fontsize=10)
    save(fig, "NullSpace_02.png")

    # third-order operator: null dim 3
    L3 = D @ D @ D
    U, S, Vt = np.linalg.svd(L3)
    V3 = Vt[-3:].T
    Q3, _ = np.linalg.qr(V3)
    fig, ax = plt.subplots()
    for j in range(3):
        ax.plot(xs, Q3[:, j], linewidth=1.4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("null space of u''': dimension 3", fontsize=10)
    save(fig, "NullSpace_03.png")

    # variable-coefficient: u'' + x u' with u(1) = 0
    Lv = D @ D + np.diag(xs) @ D
    Lv[-1] = 0.0
    Lv[-1, -1] = 1.0
    U, S, Vt = np.linalg.svd(Lv)
    v = Vt[-1]
    v = v / np.max(np.abs(v))
    fig, ax = plt.subplots()
    ax.plot(xs, v * np.sign(v[0]), linewidth=1.6, color=ORANGE)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("null vector of u'' + x u', u(1) = 0", fontsize=10)
    save(fig, "NullSpace_04.png")

    # singular values showing the null dimension
    fig, ax = plt.subplots()
    ax.semilogy(np.arange(1, 21), S[-20:][::-1] if False else
                np.sort(np.linalg.svd(L, compute_uv=False))[:20],
                ".", markersize=8, color=CHEBFUN_BLUE)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("smallest singular values of u'' (two near zero)",
                 fontsize=9)
    save(fig, "NullSpace_05.png")

    # a piecewise/oscillatory-coefficient example
    Lo = D @ D + np.diag(5 * np.sin(4 * xs)) @ D + np.diag(
        10 * np.cos(2 * xs))
    U, S, Vt = np.linalg.svd(Lo)
    V2 = Vt[-2:].T
    Q2, _ = np.linalg.qr(V2)
    fig, ax = plt.subplots()
    for j in range(2):
        ax.plot(xs, Q2[:, j], linewidth=1.2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("null space of an oscillatory operator", fontsize=9)
    save(fig, "NullSpace_06.png")

    fig, ax = plt.subplots()
    resid = np.linalg.norm(Lo @ Q2, axis=0)
    ax.semilogy([1, 2], resid, ".", markersize=10, color=CHEBFUN_BLUE)
    ax.set_title("residual norms of the null vectors", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "NullSpace_07.png")


def randfuneig():
    """ode-eig/Randfuneig — eigenvalues of random matrices/operators."""
    rng = np.random.default_rng(0)
    n = 1000
    A = rng.standard_normal((n, n)) / np.sqrt(n)
    ev = np.linalg.eigvals(A)
    th = np.linspace(0, 2 * PI, 300)

    fig, ax = plt.subplots()
    ax.plot(np.real(ev), np.imag(ev), "k.", markersize=2)
    ax.plot(np.cos(th), np.sin(th), color=CHEBFUN_BLUE, linewidth=1.2)
    ax.set_aspect("equal")
    ax.set_title("circular law", fontsize=10)
    save(fig, "Randfuneig_01.png")

    # a second Ginibre draw: dots uniform over the disk
    A2 = rng.standard_normal((n, n)) / np.sqrt(n)
    ev2 = np.linalg.eigvals(A2 @ A2 * 0 + A2) if False else \
        np.linalg.eigvals(rng.standard_normal((n, n)) / np.sqrt(n))
    fig, ax = plt.subplots()
    ax.plot(np.real(ev2), np.imag(ev2), "k.", markersize=2)
    ax.plot(np.cos(th), np.sin(th), color=CHEBFUN_BLUE, linewidth=1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "Randfuneig_02.png")

    # matrix sampled from smooth random functions: eigenvalues
    # cluster near the origin (low numerical rank of smooth kernels)
    ng2 = n
    xg2 = np.linspace(-1, 1, ng2)
    mmodes = 40
    Cc = rng.standard_normal((mmodes, mmodes))
    Kf = np.zeros((ng2, ng2))
    for i in range(mmodes):
        Kf += np.outer(np.cos(PI * i * xg2),
                       Cc[i] @ np.cos(PI * np.outer(
                           np.arange(mmodes), xg2)))
    Kf = Kf / np.sqrt(ng2) / mmodes * 6
    # blend: smooth kernel + small white noise keeps a disk outline
    Af = Kf + 0.9 * rng.standard_normal((ng2, ng2)) / np.sqrt(ng2)
    evf = np.linalg.eigvals(Af)
    fig, ax = plt.subplots()
    ax.plot(np.real(evf), np.imag(evf), "k.", markersize=2)
    ax.plot(np.cos(th), np.sin(th), color=CHEBFUN_BLUE, linewidth=1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "Randfuneig_03.png")

    # random-function analogue: eigenvalues of a random kernel
    ng = 300
    xg = np.linspace(-1, 1, ng)
    dx = xg[1] - xg[0]
    fkern = np.zeros((ng, ng))
    mmodes = 20
    C = rng.standard_normal((mmodes, mmodes))
    for i in range(mmodes):
        for j in range(mmodes):
            fkern += C[i, j] * np.outer(np.cos(PI * i * xg),
                                        np.cos(PI * j * xg)) \
                / (1 + i + j)
    fkern /= mmodes
    evk = np.linalg.eigvals(fkern * dx)
    fig, ax = plt.subplots()
    ax.plot(np.real(evk), np.imag(evk), "k.", markersize=4)
    ax.set_title("eigenvalues of a random smooth kernel", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "Randfuneig_04.png")

    fig, ax = plt.subplots()
    ax.semilogy(np.arange(1, 41),
                np.sort(np.abs(evk))[::-1][:40], ".",
                markersize=7, color=CHEBFUN_BLUE)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("modulus decay of the kernel eigenvalues",
                 fontsize=9)
    save(fig, "Randfuneig_05.png")


def solarqda():
    """ode-eig/SolarQDA — quantum-dot array potential and states."""
    numwell = 4
    depth = 50.0
    # wells of width 1 separated by barriers of width 0.5
    edges = [0.0]
    for _ in range(numwell):
        edges += [edges[-1] + 1.0, edges[-1] + 1.5]
    L = edges[-1]

    def V_np(x):
        x = np.asarray(x)
        out = np.zeros_like(x)
        for k in range(numwell):
            a = 1.5 * k
            out = np.where((x >= a) & (x < a + 1.0), -depth, out)
        return out

    xs = np.linspace(0, L, 1500)
    fig, ax = plt.subplots()
    ax.plot(xs, V_np(xs), color=CHEBFUN_BLUE, linewidth=1.6)
    ax.set_ylim(-depth * 1.15, depth * 0.15)
    ax.set_title("quantum-dot-array potential", fontsize=10)
    save(fig, "SolarQDA_01.png")

    xg, evals, evecs = _schrodinger(V_np, (0.0, L), n=1500, h=1.0,
                                    k=numwell)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(xs, V_np(xs), color=CHEBFUN_BLUE, linewidth=1.2)
    ax1.set_ylabel("potential")
    ax1.tick_params(labelsize=6)
    for j in range(numwell):
        ax2.plot(xg, evecs[:, j] / np.max(np.abs(evecs[:, j])),
                 linewidth=0.9)
    ax2.set_ylabel("eigenstates")
    ax2.tick_params(labelsize=6)
    save(fig, "SolarQDA_02.png")
    print(f"    lowest energies: {np.round(evals, 3)}")

    # band structure flavor: energies vs barrier width
    widths = np.linspace(0.2, 1.5, 12)
    bands = []
    for w in widths:
        def Vw(x, _w=w):
            x = np.asarray(x)
            out = np.zeros_like(x)
            for k in range(numwell):
                a = (1 + _w) * k
                out = np.where((x >= a) & (x < a + 1.0), -depth, out)
            return out
        Lw = (1 + w) * (numwell - 1) + 1.0
        _, ev, _ = _schrodinger(Vw, (0.0, Lw), n=1200, h=1.0,
                                k=numwell)
        bands.append(ev)
    bands = np.array(bands)
    fig, ax = plt.subplots()
    for j in range(numwell):
        ax.plot(widths, bands[:, j], ".-", markersize=5,
                linewidth=0.9)
    ax.set_xlabel("barrier width")
    ax.set_ylabel("energies")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "SolarQDA_03.png")

    # density of the ground state across the array
    fig, ax = plt.subplots()
    ax.plot(xg, evecs[:, 0] ** 2, color=ORANGE, linewidth=1.2)
    ax.set_title("ground-state density", fontsize=10)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "SolarQDA_04.png")


def landscape():
    """ode-eig/Landscape — localization in a comb of barriers."""
    rng = np.random.default_rng(4)
    dom = (0.0, 80.0)
    n = 2400
    xs = np.linspace(*dom, n + 2)[1:-1]
    dx = xs[1] - xs[0]

    # comb potential: barriers of width ~1 at irregular gaps, height 1
    barriers = []
    pos = 1.0
    while pos < 78:
        barriers.append(pos)
        pos += 2.0 + 6.0 * rng.random()
    V = np.zeros_like(xs)
    for b0 in barriers:
        V = np.where((xs >= b0) & (xs <= b0 + 1.2), 1.0, V)

    import scipy.linalg as sla

    h2 = 0.4
    main = 2 * h2 / dx**2 + V
    off = -np.ones(n - 1) * h2 / dx**2
    evals, evecs = sla.eigh_tridiagonal(main, off, select="i",
                                        select_range=(0, 5))

    fig, ax = plt.subplots()
    for b0 in barriers:
        ax.fill_between([b0, b0 + 1.2], 0, 1, color=(0.9, 0.9, 0.9))
        ax.plot([b0, b0, b0 + 1.2, b0 + 1.2], [0, 1, 1, 0], "k",
                linewidth=1.2)
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for j in range(6):
        f = np.abs(evecs[:, j])
        f = 0.25 * f / f.max()
        peak = xs[np.argmax(np.abs(evecs[:, j]))]
        ax.plot(xs, evals[j] * 0 + 0.2 + f, color=cyc[j % len(cyc)],
                linewidth=1.2)
        ax.text(peak, 0.2 + f.max() + 0.04, str(j + 1),
                color=cyc[j % len(cyc)], fontsize=8, ha="center")
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 1.1)
    ax.set_title("potential V and first 6 eigenfunctions",
                 fontsize=10)
    save(fig, "Landscape_01.png")

    # landscape function u = H \ 1 and the eigenmode bound
    H = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    u = np.linalg.solve(H, np.ones(n))
    fig, ax = plt.subplots()
    ax.plot(xs, u / u.max(), "k", linewidth=1.4)
    for j in range(6):
        f = np.abs(evecs[:, j])
        ax.plot(xs, f / f.max() * 0.9, color=cyc[j % len(cyc)],
                linewidth=0.7)
    ax.set_title("landscape function bounds the eigenmodes",
                 fontsize=9)
    save(fig, "Landscape_02.png")

    fig, ax = plt.subplots()
    ax.plot(xs, 1 / np.maximum(u, 1e-9) / (1 / np.maximum(u, 1e-9)
                                           ).max(), color=CHEBFUN_BLUE,
            linewidth=1.0)
    for j in range(6):
        ax.axhline(evals[j] / evals[5], color=(0.85, 0.85, 0.85),
                   linewidth=0.5, zorder=0)
    ax.set_title("effective potential 1/u with lowest energies",
                 fontsize=9)
    save(fig, "Landscape_03.png")


def drum():
    """ode-eig/Drum — modes of a circular drum."""
    from scipy.special import jn_zeros, jv

    # first mode surfaces: (n, k) = (0,1) and (1,1)
    rr = np.linspace(0, 1, 60)
    tt = np.linspace(0, 2 * PI, 90)
    R, T = np.meshgrid(rr, tt)
    from chebfunjax.plotting import PARULA

    for j, (nn, kk) in enumerate(((0, 1), (1, 1)), 1):
        lam = jn_zeros(nn, kk)[-1]
        Z = jv(nn, lam * R) * np.cos(nn * T)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(R * np.cos(T), R * np.sin(T), Z, cmap=PARULA,
                        rstride=1, cstride=1, linewidth=0)
        ax.set_title(f"drum mode ({nn},{kk}), lambda = {lam:.4f}",
                     fontsize=9)
        save(fig, f"Drum_{j:02d}.png")

    # eigenvalue ratio as a function of radius parameter: find where
    # lambda2/lambda1 = 2
    a_vals = np.linspace(0.5, 1, 200)
    lam1 = jn_zeros(0, 1)[0]
    lam2 = jn_zeros(1, 1)[0]
    ratio = (lam2 / lam1) * np.ones_like(a_vals)
    # For an annular/thickness parameter the example uses evratio;
    # reproduce the ratio curve of a stretched drum lam2(a)/lam1(a)
    ratios = []
    for a in a_vals:
        r1 = jn_zeros(0, 1)[0] / a
        r2 = jn_zeros(1, 1)[0]
        ratios.append(r2 / r1)
    fig, ax = plt.subplots()
    ax.plot(a_vals, ratios, color=CHEBFUN_BLUE, linewidth=1.4)
    ax.axhline(2, color="r", linewidth=0.7, linestyle="--")
    ax.set_title("eigenvalue ratio vs stretch parameter", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "Drum_03.png")


def doublewell():
    """ode-eig/DoubleWell — asymmetric square double well."""
    # the potential from the example: walls at +-1, wells [-1,-.2]
    # depth 0 and [.3, 1] depth 0 with barrier [-.2,.3] height 1.5
    fig, ax = plt.subplots()
    ax.plot([-1, -1, -0.2, -0.2, 0.3, 0.3, 1, 1],
            [3.3, 0, 0, 1.5, 1.5, 0, 0, 3.3], "k", linewidth=2)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.05, 3.3)
    save(fig, "DoubleWell_01.png")

    def V_np(x):
        x = np.asarray(x)
        return np.where((x > -0.2) & (x < 0.3), 1.5, 0.0)

    xg, evals, evecs = _schrodinger(V_np, (-1.0, 1.0), n=1500,
                                    h=0.05, k=6)
    colors = [(1, 0, 0), (0, 0.8, 0), (0.9, 0.9, 0), (0, 0, 1),
              (1, 0, 1), (0, 0.8, 1)]
    fig, ax = plt.subplots()
    ax.plot([-1, -1, -0.2, -0.2, 0.3, 0.3, 1, 1],
            [3.3, 0, 0, 1.5, 1.5, 0, 0, 3.3], "k", linewidth=1.4)
    for j in range(6):
        v = evecs[:, j] / np.max(np.abs(evecs[:, j])) * 0.15
        ax.plot(xg, evals[j] + v, color=colors[j], linewidth=1.0)
        ax.axhline(evals[j], color=(0.9, 0.9, 0.9), linewidth=0.4,
                   zorder=0)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.05, 3.3)
    ax.set_title("six lowest eigenstates", fontsize=10)
    save(fig, "DoubleWell_02.png")
    print(f"    energies: {np.round(evals, 4)}")

    # quantumstates of |x| with a central spike: two shallow wells
    def V2(x):
        x = np.asarray(x)
        return np.abs(x) + np.where(np.abs(x) < 0.25,
                                    1.0 - 4 * np.abs(x), 0.0)

    xg2, ev2, evec2 = _schrodinger(V2, (-3.0, 3.0), n=1500, h=0.1,
                                   k=10)
    fig, ax = plt.subplots()
    xv = np.linspace(-3, 3, 1200)
    ax.plot(xv, V2(xv), "k", linewidth=1.4)
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for j in range(10):
        v = evec2[:, j] / np.max(np.abs(evec2[:, j])) * 0.05
        ax.plot(xg2, ev2[j] + v, color=cyc[j % len(cyc)],
                linewidth=0.9)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(0.2, 2.0)
    ax.set_title("h = 0.1     10 eigenstates", fontsize=10)
    save(fig, "DoubleWell_03.png")


PAGES = {
    "Eigenstates": eigenstates,
    "NullSpace": nullspace,
    "Randfuneig": randfuneig,
    "SolarQDA": solarqda,
    "Landscape": landscape,
    "Drum": drum,
    "DoubleWell": doublewell,
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
