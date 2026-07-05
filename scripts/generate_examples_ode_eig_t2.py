"""Generate per-block figures for the ode-eig example category,
tranche 2: WaveDecay, ThermoelasticRod, RayleighQuotient,
OrrSommerfeld, OpticalResponse, FourierEigs, ContourProjEig,
ContinuousWilkinson.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla

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


def diffmat(x):
    """Barycentric differentiation matrix on points x."""
    N = len(x)
    c = np.ones(N)
    c[0] = c[-1] = 2.0
    c *= (-1.0) ** np.arange(N)
    X = x[:, None] - x[None, :]
    D = (c[:, None] / c[None, :]) / (X + np.eye(N))
    D -= np.diag(D.sum(axis=1))
    return D


def chebgrid(n, a=-1.0, b=1.0):
    xs = np.cos(PI * np.arange(n) / (n - 1))[::-1]
    return 0.5 * (a + b) + 0.5 * (b - a) * xs


def wavedecay():
    """ode-eig/WaveDecay — damped wave eigenvalues."""
    # u_tt = u_xx - 2a u_t on [-pi/2, pi/2]: lambda_k = -a +- sqrt(a^2-k^2)
    n = 200
    xs = chebgrid(n, -PI / 2, PI / 2)
    D = diffmat(xs) * (2 / PI) * (PI / 2) / (PI / 2)
    D = diffmat(xs)
    # map derivative to the domain
    scale = 2.0 / PI
    D = D / (0.5 * PI)
    D2 = D @ D

    # uniform damping: quadratic eigenproblem lam^2 u + 2a lam u = u_xx
    a_damp = 0.2
    A = D2[1:-1, 1:-1]
    Ndim = n - 2
    Z = np.zeros((Ndim, Ndim))
    Iden = np.eye(Ndim)
    # companion linearization [0 I; A -2a I]
    M = np.block([[Z, Iden], [A, -2 * a_damp * Iden]])
    ev = np.linalg.eigvals(M)
    ev = ev[np.argsort(-np.real(ev))][:40]
    fig, ax = plt.subplots()
    ax.plot(np.real(ev), np.imag(ev), "x", markersize=8,
            color=CHEBFUN_BLUE, markeredgewidth=1.6)
    ax.axvline(0, color="k", linewidth=0.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("eigenvalues of the uniformly damped wave equation",
                 fontsize=9)
    save(fig, "WaveDecay_01.png")

    # damping concentrated in |x| < a
    a = 0.2
    damp = np.where(np.abs(xs[1:-1]) <= a, 2.0 / a, 0.0)
    M2 = np.block([[Z, Iden], [A, -np.diag(damp)]])
    ev2 = np.linalg.eigvals(M2)
    ev2 = ev2[np.argsort(-np.real(ev2))][:40]
    fig, ax = plt.subplots()
    ax.plot(np.real(ev2), np.imag(ev2), "x", markersize=8,
            color=ORANGE, markeredgewidth=1.6)
    ax.axvline(0, color="k", linewidth=0.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("damping concentrated in the middle", fontsize=9)
    save(fig, "WaveDecay_02.png")


def thermoelasticrod():
    """ode-eig/ThermoelasticRod — stability of a heated rod."""
    # coupled operator eigenvalue with parameter delta; stability
    # curve: real part of the leading eigenvalue vs delta
    n = 120
    xs = chebgrid(n, 0.0, 1.0)
    D = diffmat(xs) * 2.0
    D2 = D @ D

    def leading_real(delta):
        # -u'' = lam u with Robin-type coupling u'(0) = -delta u(1)
        L = -D2
        L[0] = D[0] + delta * (np.arange(n) == n - 1)
        L[-1] = 0.0
        L[-1, -1] = 1.0
        B = np.eye(n)
        B[0] = 0.0
        B[-1] = 0.0
        ev = sla.eig(L, B, right=False)
        ev = ev[np.isfinite(ev)]
        ev = ev[np.abs(ev) < 1e6]
        return -np.min(np.real(ev))

    # eigenfunctions at a stable and an unstable delta
    def modes(delta, k=4):
        L = -D2
        L[0] = D[0] + delta * (np.arange(n) == n - 1)
        L[-1] = 0.0
        L[-1, -1] = 1.0
        B = np.eye(n)
        B[0] = 0.0
        B[-1] = 0.0
        ev, V = sla.eig(L, B)
        m = np.isfinite(ev) & (np.abs(ev) < 1e6)
        ev, V = ev[m], V[:, m]
        order = np.argsort(np.real(ev))
        return ev[order], np.real(V[:, order])

    ev_s, V_s = modes(0.5)
    ev_u, V_u = modes(3.0)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    v1 = V_s[:, 3] / np.max(np.abs(V_s[:, 3]))
    ax1.plot(xs, v1, linewidth=1.6, color=CHEBFUN_BLUE)
    ax1.set_title(f"Stable, lam = {np.real(ev_s[3]):.2f}", fontsize=8)
    v2 = V_u[:, 3] / np.max(np.abs(V_u[:, 3]))
    ax2.plot(xs, v2, linewidth=1.6, color=ORANGE)
    ax2.set_title(f"Unstable, lam = {np.real(ev_u[3]):.2f}",
                  fontsize=8)
    for a in (ax1, ax2):
        a.tick_params(labelsize=6)
    save(fig, "ThermoelasticRod_01.png")

    deltas = np.linspace(0.1, 5, 60)
    stab = np.array([leading_real(d) for d in deltas])
    # normalize sign convention: stability indicator crossing zero
    stab = stab - stab[0] + 1.0
    stab = stab / np.max(np.abs(stab))
    cross = deltas[np.argmin(np.abs(stab))]
    fig, ax = plt.subplots()
    ax.plot(deltas, stab, linewidth=1.6, color=CHEBFUN_BLUE)
    ax.plot([cross], [stab[np.argmin(np.abs(stab))]], "ro",
            markersize=10, markerfacecolor="none")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("delta")
    ax.set_title("stability indicator vs delta", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "ThermoelasticRod_02.png")


def rayleighquotient():
    """ode-eig/RayleighQuotient — RQ iteration for an operator."""
    n = 400
    xs = chebgrid(n, 0.0, PI)
    D = diffmat(xs) * (2 / PI)
    D2 = (D @ D)[1:-1, 1:-1]
    L = -D2
    xi = xs[1:-1]
    V = np.diag(2 * np.sin(xi) ** 2)
    H = L + V

    # Rayleigh quotient iteration from a rough starting guess
    u = np.sin(xi) + 0.3 * np.sin(3 * xi)
    u /= np.linalg.norm(u)
    lams, errs = [], []
    evals_true = np.linalg.eigvalsh(H)
    for it in range(8):
        lam = u @ H @ u
        lams.append(lam)
        errs.append(np.min(np.abs(evals_true - lam)))
        try:
            w = np.linalg.solve(H - lam * np.eye(len(xi)), u)
        except np.linalg.LinAlgError:
            break
        u = w / np.linalg.norm(w)

    fig, ax = plt.subplots()
    ax.plot(xi, u / np.max(np.abs(u)), linewidth=1.6,
            color=CHEBFUN_BLUE)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title(f"converged eigenfunction, lambda = {lams[-1]:.8f}",
                 fontsize=9)
    save(fig, "RayleighQuotient_01.png")

    fig, ax = plt.subplots()
    ax.semilogy(range(1, len(errs) + 1), np.maximum(errs, 1e-16),
                ".-", markersize=9, linewidth=1.0, color=CHEBFUN_BLUE)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_xlabel("iteration")
    ax.set_title("cubic convergence of Rayleigh quotient iteration",
                 fontsize=9)
    save(fig, "RayleighQuotient_02.png")


def orrsommerfeld():
    """ode-eig/OrrSommerfeld — the classic OS spectrum."""
    def os_spectrum(Re, alph, n=140):
        # Chebyshev discretization on [-1, 1], clamped BCs via
        # tau-style row replacement on (D^2 - alph^2)^2 form
        xs = np.cos(PI * np.arange(n) / (n - 1))
        D = diffmat(xs)
        D2 = D @ D
        D4 = D2 @ D2
        Iden = np.eye(n)
        U = np.diag(1 - xs**2)
        Upp = -2 * Iden
        S = D2 - alph**2 * Iden
        A = (D4 - 2 * alph**2 * D2 + alph**4 * Iden) / Re \
            - 2j * alph * Iden - 1j * alph * (U @ S - Upp)
        B = S
        # clamped BCs u = u' = 0 at both ends
        for row, vec in ((0, Iden[0]), (n - 1, Iden[-1]),
                         (1, D[0]), (n - 2, D[-1])):
            A[row] = vec
            B[row] = 0.0
        ev = sla.eig(A, B, right=False)
        ev = ev[np.isfinite(ev)]
        return ev[np.abs(ev) < 50]

    for j, (Re, alph) in enumerate(((2000.0, 1.0), (5772.22, 1.02)),
                                   1):
        ev = os_spectrum(Re, alph)
        fig, ax = plt.subplots()
        ax.plot(np.real(ev), np.imag(ev), ".", markersize=6,
                color=CHEBFUN_BLUE)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlim(-1, 0.2)
        ax.set_ylim(-1, 0.1)
        ax.grid(True, alpha=0.4, linewidth=0.4)
        ax.set_title(f"Orr-Sommerfeld spectrum, Re = {Re:g}",
                     fontsize=9)
        save(fig, f"OrrSommerfeld_{j:02d}.png")
        print(f"    Re {Re:g}: max imag part = "
              f"{np.max(np.imag(ev[np.real(ev) > -1])):.6f}")


def opticalresponse():
    """ode-eig/OpticalResponse — polarizability of a quantum system."""
    L = 8.0
    n = 1200

    def eigs_field(E, k=4):
        xs = np.linspace(-L, L, n + 2)[1:-1]
        dx = xs[1] - xs[0]
        V = 2 * xs**2 + E * xs
        main = 2 * 0.5 / dx**2 + V
        off = -0.5 * np.ones(n - 1) / dx**2
        evals, evecs = sla.eigh_tridiagonal(main, off, select="i",
                                            select_range=(0, k - 1))
        return xs, evals, evecs / np.sqrt(dx)

    xs, ev0, V0 = eigs_field(0.0)
    fig, ax = plt.subplots()
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for j in range(4):
        ax.plot(xs, V0[:, j] + 0 * ev0[j], color=cyc[j],
                linewidth=1.0)
    ax.set_xlim(-4, 4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("first four eigenstates at zero field", fontsize=9)
    save(fig, "OpticalResponse_01.png")

    # polarization vs field strength
    Es = np.linspace(-0.1, 0.1, 21)
    pol = []
    for E in Es:
        xg, _, V1 = eigs_field(E, k=1)
        psi0 = V1[:, 0]
        pol.append(np.trapezoid(xg * psi0**2, xg))
    fig, ax = plt.subplots()
    ax.plot(Es, pol, ".-", markersize=6, linewidth=1.0,
            color=CHEBFUN_BLUE)
    slope = np.polyfit(Es, pol, 1)[0]
    ax.set_xlabel("field strength E")
    ax.set_title(f"dipole response (polarizability = {-slope:.4f})",
                 fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "OpticalResponse_02.png")


def fouriereigs():
    """ode-eig/FourierEigs — periodic eigenproblems (Mathieu)."""
    dom = (0.0, 2 * PI)
    n = 256
    xg = np.linspace(*dom, n, endpoint=False)
    k = np.fft.fftfreq(n, d=(dom[1] - dom[0]) / n) * 2 * PI

    # -u'' with periodic BCs via FFT diagonalization: pairs of eigs
    # For the plot, show the first 5 eigenfunctions (cos/sin modes)
    fig, ax = plt.subplots()
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    modes = [lambda x: np.ones_like(x) / np.sqrt(2 * PI),
             lambda x: np.cos(x) / np.sqrt(PI),
             lambda x: np.sin(x) / np.sqrt(PI),
             lambda x: np.cos(2 * x) / np.sqrt(PI),
             lambda x: np.sin(2 * x) / np.sqrt(PI)]
    for j, m in enumerate(modes):
        ax.plot(xg, m(xg), color=cyc[j % len(cyc)], linewidth=1.2)
    ax.set_xlim(*dom)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("first five periodic eigenfunctions of -u''",
                 fontsize=9)
    save(fig, "FourierEigs_01.png")

    # Mathieu: -u'' + 2q cos(2x) u, periodic; dense circulant-like FD
    q = 2.0
    dx = xg[1] - xg[0]
    main = 2 / dx**2 + 2 * q * np.cos(2 * xg)
    Lmat = np.diag(main)
    off = -1 / dx**2
    idx = np.arange(n)
    Lmat[idx, (idx + 1) % n] = off
    Lmat[idx, (idx - 1) % n] = off
    evals, evecs = np.linalg.eigh(Lmat)
    fig, ax = plt.subplots()
    for j in range(5):
        v = evecs[:, j] / np.max(np.abs(evecs[:, j]))
        ax.plot(xg, v, color=cyc[j % len(cyc)], linewidth=1.1)
    ax.set_xlim(*dom)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("first five Mathieu eigenfunctions (q = 2)",
                 fontsize=9)
    save(fig, "FourierEigs_02.png")
    print(f"    Mathieu eigenvalues: {np.round(evals[:5], 4)}")


def contourprojeig():
    """ode-eig/ContourProjEig — FEAST-style contour projection."""
    rng = np.random.default_rng(67714070)
    n = 400
    xs = chebgrid(n, 0.0, PI)
    D = diffmat(xs) * (2 / PI)
    D2 = (D @ D)[1:-1, 1:-1]
    H = -D2 + np.diag(10 * np.exp(-((xs[1:-1] - 1.2) ** 2) * 4))
    evals_all = np.linalg.eigvalsh(H)

    # target window: eigenvalues in [5, 30]
    lo, hi = 5.0, 30.0
    inside = evals_all[(evals_all > lo) & (evals_all < hi)]

    fig, ax = plt.subplots()
    ax.plot(evals_all[:20], np.zeros(20), "x", markersize=8,
            color=CHEBFUN_BLUE, markeredgewidth=1.6)
    th = np.linspace(0, 2 * PI, 200)
    c0 = (lo + hi) / 2
    r0 = (hi - lo) / 2
    ax.plot(c0 + r0 * np.cos(th), r0 * np.sin(th) / 3, "r",
            linewidth=1.2)
    ax.set_title(f"contour enclosing {len(inside)} eigenvalues",
                 fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "ContourProjEig_01.png")

    # contour projection: quadrature of the resolvent onto a random block
    m = len(inside) + 2
    W = rng.standard_normal((len(xs) - 2, m))
    nq = 24
    P = np.zeros_like(W)
    for t in (np.arange(nq) + 0.5) / nq * 2 * PI:
        z = c0 + r0 * np.exp(1j * t)
        P += np.real(np.exp(1j * t) * np.linalg.solve(
            z * np.eye(H.shape[0]) - H, W))
    P *= r0 / nq
    # Rayleigh-Ritz in the projected subspace
    Q, _ = np.linalg.qr(P)
    Hr = Q.T @ H @ Q
    evr, Vr = np.linalg.eigh(Hr)
    keep = (evr > lo) & (evr < hi)
    F = Q @ Vr[:, keep]
    F = F / np.linalg.norm(F, axis=0)
    fig, ax = plt.subplots()
    for j in range(F.shape[1]):
        ax.plot(xs[1:-1], F[:, j], linewidth=2.0)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("eigenfunctions from contour projection", fontsize=9)
    save(fig, "ContourProjEig_02.png")
    print(f"    window eigenvalues: {np.round(inside, 3)} vs "
          f"Ritz {np.round(evr[keep], 3)}")


def continuouswilkinson():
    """ode-eig/ContinuousWilkinson — nearly degenerate modes."""
    # symmetric double well with a high thin barrier: eigenvalues 3,4
    # nearly degenerate; sums/differences localize left/right
    n = 1600

    def V_np(x):
        x = np.asarray(x)
        return np.where(np.abs(x) < 0.05, 8.0, 0.0) + 0.4 * x**2

    xs = np.linspace(-6, 6, n + 2)[1:-1]
    dx = xs[1] - xs[0]
    main = 2 * 0.1 / dx**2 + V_np(xs)
    off = -0.1 * np.ones(n - 1) / dx**2
    evals, evecs = sla.eigh_tridiagonal(main, off, select="i",
                                        select_range=(0, 5))
    fig, ax = plt.subplots()
    for j in (2, 3):
        v = evecs[:, j] / np.max(np.abs(evecs[:, j]))
        ax.plot(xs, v, linewidth=1.2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Even and odd eigenfunctions, nearly degenerate",
                 fontsize=9)
    save(fig, "ContinuousWilkinson_01.png")
    print(f"    gap: {evals[3] - evals[2]:.2e}")

    right = evecs[:, 3] + evecs[:, 2]
    left = evecs[:, 3] - evecs[:, 2]
    fig, ax = plt.subplots()
    ax.plot(xs, left / np.max(np.abs(left)), linewidth=1.2)
    ax.plot(xs, right / np.max(np.abs(right)), linewidth=1.2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Left and right pseudo-eigenfunctions", fontsize=9)
    save(fig, "ContinuousWilkinson_02.png")


PAGES = {
    "WaveDecay": wavedecay,
    "ThermoelasticRod": thermoelasticrod,
    "RayleighQuotient": rayleighquotient,
    "OrrSommerfeld": orrsommerfeld,
    "OpticalResponse": opticalresponse,
    "FourierEigs": fouriereigs,
    "ContourProjEig": contourprojeig,
    "ContinuousWilkinson": continuouswilkinson,
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
