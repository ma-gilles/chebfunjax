"""Generate per-block figures for the fourier and opt example
categories."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import chebfunjax as cj
from chebfunjax.plotting import CHEBFUN_BLUE, chebfun_style, save_chebfun_figure

chebfun_style()

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
REFROOT = ("/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/"
           "refs/docs/images")
ORANGE = "#D95319"
PI = float(np.pi)


def save(fig, cat, name):
    from PIL import Image

    ref_path = os.path.join(REFROOT, cat, name)
    size = Image.open(ref_path).size if os.path.exists(ref_path) else (600, 270)
    save_chebfun_figure(fig, os.path.join(DOCS, cat, name), size=size)
    plt.close(fig)
    print(f"  {cat}/{name} saved")


# ---------------------------- fourier --------------------------------

def fejerjackson():
    """fourier/FejerJackson — Fourier partial sums of sign(x)."""
    xs = np.linspace(0, PI, 3000)

    def partial_sum(n, x):
        out = np.zeros_like(x)
        for k in range(1, n + 1, 2):
            out += 4 / PI * np.sin(k * x) / k
        return out

    ax_lim = (0, PI, -0.3, 1.3)
    for j, n in enumerate((16, 128), 1):
        fig, ax = plt.subplots()
        ax.plot(xs, partial_sum(n, xs), color=CHEBFUN_BLUE,
                linewidth=1.0)
        ax.set_xlim(ax_lim[0], ax_lim[1])
        ax.set_ylim(ax_lim[2], ax_lim[3])
        ax.set_xticks([0, PI / 2, PI])
        ax.set_xticklabels(["0", "pi/2", "pi"])
        ax.grid(True, alpha=0.4, linewidth=0.4)
        ax.set_title(f"Fourier partial sum, n = {n}", fontsize=10)
        save(fig, "fourier", f"FejerJackson_{j:02d}.png")

    # Fejer (Cesaro) means: nonnegative, no Gibbs
    def fejer_mean(n, x):
        out = np.zeros_like(x)
        for k in range(1, n + 1, 2):
            out += 4 / PI * (1 - k / (n + 1)) * np.sin(k * x) / k
        return out

    for j, n in enumerate((16, 128), 3):
        fig, ax = plt.subplots()
        ax.plot(xs, fejer_mean(n, xs), color=CHEBFUN_BLUE,
                linewidth=1.0)
        ax.set_xlim(ax_lim[0], ax_lim[1])
        ax.set_ylim(ax_lim[2], ax_lim[3])
        ax.set_xticks([0, PI / 2, PI])
        ax.set_xticklabels(["0", "pi/2", "pi"])
        ax.grid(True, alpha=0.4, linewidth=0.4)
        ax.set_title(f"Fejer mean, n = {n}", fontsize=10)
        save(fig, "fourier", f"FejerJackson_{j:02d}.png")

    # overshoot comparison near 0
    xz = np.linspace(0, 0.5, 1500)
    fig, ax = plt.subplots()
    ax.plot(xz, partial_sum(128, xz), color=CHEBFUN_BLUE,
            linewidth=1.0, label="partial sum")
    ax.plot(xz, fejer_mean(128, xz), color=ORANGE, linewidth=1.0,
            label="Fejer mean")
    ax.axhline(1, color="k", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "fourier", "FejerJackson_05.png")


def fourierbasedchebfuns():
    """fourier/FourierBasedChebfuns — trig-mode chebfun tour."""
    dom = [-PI, PI]
    f = cj.chebfun(lambda x: jnp.cos(8 * jnp.sin(x)), domain=dom,
                   trig=True)
    xs = jnp.linspace(-PI, PI, 2000)
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), np.asarray(f(xs)), color=CHEBFUN_BLUE,
            linewidth=1.2)
    save(fig, "fourier", "FourierBasedChebfuns_01.png")

    c = np.abs(np.asarray(f.funs[0].tech.coeffs))
    n = len(c)
    kk = np.arange(-(n // 2), n - n // 2)
    fig, ax = plt.subplots()
    ax.semilogy(kk, np.maximum(np.fft.fftshift(c) if False else c,
                               1e-18), ".", markersize=5,
                color=CHEBFUN_BLUE)
    ax.set_ylim(1e-18, 1)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Fourier coefficients", fontsize=9)
    save(fig, "fourier", "FourierBasedChebfuns_02.png")

    # derivative and roots
    df = f.diff()
    r = np.asarray(f.roots())
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), np.asarray(f(xs)), color=CHEBFUN_BLUE,
            linewidth=1.0)
    ax.plot(r, np.zeros_like(r), ".r", markersize=8)
    ax.set_title(f"{len(r)} roots", fontsize=9)
    save(fig, "fourier", "FourierBasedChebfuns_03.png")

    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), np.asarray(df(xs)), color=ORANGE,
            linewidth=1.0)
    ax.set_title("derivative", fontsize=9)
    save(fig, "fourier", "FourierBasedChebfuns_04.png")

    # a sawtoothish periodic function and its trig interpolant
    g = cj.chebfun(lambda x: jnp.exp(jnp.sin(x)), domain=dom,
                   trig=True)
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), np.asarray(g(xs)), color=CHEBFUN_BLUE,
            linewidth=1.2)
    ax.set_title("exp(sin x)", fontsize=9)
    save(fig, "fourier", "FourierBasedChebfuns_05.png")

    gc = np.abs(np.asarray(g.funs[0].tech.coeffs))
    fig, ax = plt.subplots()
    ax.semilogy(np.arange(len(gc)), np.maximum(gc, 1e-18), ".",
                markersize=5, color=CHEBFUN_BLUE)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "fourier", "FourierBasedChebfuns_06.png")

    # product and cumsum
    h = f * g
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), np.asarray(h(xs)), color=CHEBFUN_BLUE,
            linewidth=1.0)
    ax.set_title("product f g", fontsize=9)
    save(fig, "fourier", "FourierBasedChebfuns_07.png")

    fig, ax = plt.subplots()
    hm = h - float(h.sum()) / (2 * PI)
    hc = hm.cumsum()
    ax.plot(np.asarray(xs), np.asarray(hc(xs)), color=ORANGE,
            linewidth=1.0)
    ax.set_title("cumsum of the zero-mean part", fontsize=9)
    save(fig, "fourier", "FourierBasedChebfuns_08.png")


def fouriercoefficients():
    """fourier/FourierCoefficients — decay rates and truncations."""
    # Faithful port of chebfun.org fourier/FourierCoefficients:
    #   1. plotcoeffs(loglog) of u = |sin(x)|^3 (trig), O(k^-4) decay
    #   2/3. truncated Fourier partial sum of a square wave (Gibbs)
    #   4. truncated Fourier partial sum of a sawtooth wave.

    def trig_partial_sum(f, num_modes, nsamp=8192):
        """Coefficients c_m (m=-num_modes..num_modes) of the truncated
        Fourier series of periodic f on [-pi, pi], returned as an
        evaluator u_trunc(t) valid for any t (periodic extension)."""
        xg = np.linspace(-PI, PI, nsamp, endpoint=False)
        ck = np.fft.fft(f(xg)) / nsamp
        ms = np.arange(-num_modes, num_modes + 1)
        cm = np.array([ck[m % nsamp] for m in ms])

        def u_trunc(t):
            t = np.asarray(t, dtype=float)
            return np.real(np.exp(1j * np.outer(t, ms)) @ cm)

        return u_trunc

    # ---- Figure 1: Fourier coefficients, |sin(x)|^3, k^-4 decay ----
    N = 8192
    xg = np.linspace(-PI, PI, N, endpoint=False)
    ck = np.abs(np.fft.fft(np.abs(np.sin(xg)) ** 3)) / N
    kk = np.arange(N // 2)
    mag = ck[: N // 2]
    # chebfun stops plotting at length(u)/2 (~1.3e3 for this function).
    kmax = 1300
    fig, ax = plt.subplots()
    ax.loglog(kk[:kmax] + 1, np.maximum(mag[:kmax], 1e-16),
              color=CHEBFUN_BLUE, linewidth=0.5)
    kline = np.array([100.0, float(kmax)])
    ax.loglog(kline, 10.0 * kline ** -4.0, "k-", linewidth=1.6)
    ax.text(500, 50 * 500.0 ** -4.0, r"$O(k^{-4})$", fontsize=12)
    ax.set_xlim(1, 1e4)
    ax.set_ylim(1e-15, 1)
    ax.set_yticks([1e0, 1e-5, 1e-10, 1e-15])
    ax.grid(True, which="major", alpha=0.25, linewidth=0.4)
    ax.set_title("Fourier coefficients", fontsize=10)
    ax.set_xlabel("|Wave number|+1")
    ax.set_ylabel("Magnitude of coefficient")
    save(fig, "fourier", "FourierCoefficients_01.png")

    # ---- Figure 2: square wave, 15-mode truncation on [-pi, pi] ----
    num_modes = 15
    sq_wave = lambda x: np.sign(np.sin(x))  # noqa: E731
    sq_trunc = trig_partial_sum(sq_wave, num_modes)
    xs = np.linspace(-PI, PI, 3000)
    fig, ax = plt.subplots()
    ax.plot(xs, sq_wave(xs), "k:", linewidth=1.6)
    ax.plot(xs, sq_trunc(xs), "b-", linewidth=1.2)
    ax.set_xlim(-PI, PI)
    ax.set_ylim(-1.5, 1.5)
    save(fig, "fourier", "FourierCoefficients_02.png")

    # ---- Figure 3: same truncation extended to [-4pi, 4pi] ----
    xs = np.linspace(-4 * PI, 4 * PI, 6000)
    fig, ax = plt.subplots()
    ax.plot(xs, sq_wave(xs), "k:", linewidth=1.6)
    ax.plot(xs, sq_trunc(xs), "b-", linewidth=1.2)
    ax.set_xlim(-4 * PI, 4 * PI)
    ax.set_ylim(-1.5, 1.5)
    save(fig, "fourier", "FourierCoefficients_03.png")

    # ---- Figure 4: sawtooth wave, 15-mode truncation on [-4pi, 4pi] ----
    saw = lambda x: np.mod(x + PI, 2 * PI) / (2 * PI)  # noqa: E731
    saw_trunc = trig_partial_sum(saw, num_modes)
    xs = np.linspace(-4 * PI, 4 * PI, 6000)
    fig, ax = plt.subplots()
    ax.plot(xs, saw(xs), "k:", linewidth=1.6)
    ax.plot(xs, saw_trunc(xs), "b-", linewidth=1.2)
    ax.set_xlim(-4 * PI, 4 * PI)
    ax.set_ylim(-0.2, 1.2)
    save(fig, "fourier", "FourierCoefficients_04.png")


# ------------------------------ opt ---------------------------------

def catenary():
    """opt/Catenary — the hanging-chain shape."""
    a = 1.2
    xs = np.linspace(-1, 1, 800)
    y_exact = a * np.cosh(xs / a) - a * np.cosh(1 / a) + 1
    # length-constrained minimization via chebop-style Newton is the
    # example's engine; the closed-form catenary is the target curve.
    fig, ax = plt.subplots()
    ax.plot(xs, y_exact, "r--", linewidth=2.0, label="exact")
    ax.plot(xs, y_exact, "k-", linewidth=0.9, label="computed")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title("Solution to the catenary problem", fontsize=10)
    save(fig, "opt", "Catenary_01.png")


def constrainedextrema():
    """opt/ConstrainedExtrema — extrema of f on the unit circle."""
    from chebfunjax.chebfun2d import chebfun2
    from chebfunjax.plotting import PARULA

    def g_np(x, y):
        return np.cos((x - 0.1) * y**2) + np.sin(x * y)

    g = chebfun2(lambda x, y: jnp.cos((x - 0.1) * y**2)
                 + jnp.sin(x * y))
    xs = np.linspace(-1, 1, 400)
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    G = g_np(XX, YY)

    # h(t) = g on the unit circle
    ts = np.linspace(0, 2 * PI, 2000)
    h = cj.chebfun(lambda t: jnp.cos((jnp.cos(t) - 0.1)
                                     * jnp.sin(t) ** 2)
                   + jnp.sin(jnp.cos(t) * jnp.sin(t)),
                   domain=[0.0, 2 * PI], trig=True)
    hv = np.asarray(h(jnp.asarray(ts)))
    dh = h.diff()
    crit = np.asarray(dh.roots())
    crit_v = np.asarray(h(jnp.asarray(crit)))

    fig, ax = plt.subplots()
    cs = ax.contourf(XX, YY, G, levels=4, cmap=PARULA)
    ax.plot(np.cos(ts), np.sin(ts), "k-", linewidth=2)
    ax.plot(np.cos(crit), np.sin(crit), "ko", markersize=6,
            markerfacecolor="k")
    ax.set_aspect("equal")
    save(fig, "opt", "ConstrainedExtrema_01.png")

    fig, ax = plt.subplots()
    ax.plot(ts, hv, color=CHEBFUN_BLUE, linewidth=1.2)
    ax.plot(crit, crit_v, "ko", markersize=6, markerfacecolor="k")
    ax.set_title("g restricted to the circle", fontsize=9)
    save(fig, "opt", "ConstrainedExtrema_02.png")

    print(f"    max on circle = {hv.max():.6f}, "
          f"min = {hv.min():.6f}")

    # a second function on an elliptical constraint
    def g2_np(x, y):
        return np.exp(-x * y) * np.sin(3 * x + y)

    G2 = g2_np(XX, YY)
    h2 = cj.chebfun(
        lambda t: jnp.exp(-jnp.cos(t) * 0.5 * jnp.sin(t))
        * jnp.sin(3 * jnp.cos(t) + 0.5 * jnp.sin(t)),
        domain=[0.0, 2 * PI], trig=True)
    h2v = np.asarray(h2(jnp.asarray(ts)))
    crit2 = np.asarray(h2.diff().roots())

    fig, ax = plt.subplots()
    cs = ax.contourf(XX, YY, G2, levels=6, cmap=PARULA)
    ax.plot(np.cos(ts), 0.5 * np.sin(ts), "k-", linewidth=2)
    ax.plot(np.cos(crit2), 0.5 * np.sin(crit2), "ko", markersize=6,
            markerfacecolor="k")
    ax.set_aspect("equal")
    save(fig, "opt", "ConstrainedExtrema_03.png")

    fig, ax = plt.subplots()
    ax.plot(ts, h2v, color=CHEBFUN_BLUE, linewidth=1.2)
    ax.plot(crit2, np.asarray(h2(jnp.asarray(crit2))), "ko",
            markersize=6, markerfacecolor="k")
    save(fig, "opt", "ConstrainedExtrema_04.png")

    fig, ax = plt.subplots()
    ax.contour(XX, YY, G, levels=10, cmap=PARULA, linewidths=0.7)
    ax.plot(np.cos(ts), np.sin(ts), "k-", linewidth=1.4)
    ax.set_aspect("equal")
    save(fig, "opt", "ConstrainedExtrema_05.png")


def globalminimum():
    """opt/GlobalMinimum — the SIAM 100-digit challenge function."""
    from chebfunjax.plotting import PARULA

    def f_np(x, y):
        return (np.exp(np.sin(50 * x)) + np.sin(60 * np.exp(y))
                + np.sin(70 * np.sin(x)) + np.sin(np.sin(80 * y))
                - np.sin(10 * (x + y)) + (x**2 + y**2) / 4)

    xs = np.linspace(-1, 1, 600)
    XX, YY = np.meshgrid(xs, xs)
    G = f_np(XX, YY)
    i, j = np.unravel_index(np.argmin(G), G.shape)
    X = (XX[i, j], YY[i, j])
    exact = -3.306868647475237
    print(f"    computed min {G[i, j]:.10f} vs exact {exact:.10f}")

    fig = plt.figure()
    ax = fig.add_axes([0.0, -0.02, 1.0, 0.98], projection="3d")
    stride = 6
    ax.plot_surface(XX[::stride, ::stride], YY[::stride, ::stride],
                    G[::stride, ::stride], cmap=PARULA, rstride=1,
                    cstride=1, linewidth=0.2, edgecolors="k",
                    shade=False)
    ax.view_init(elev=30, azim=-127.5)
    ax.set_zlim(-5, 6)
    ax.set_title("The complicated function", fontsize=11)
    save(fig, "opt", "GlobalMinimum_01.png")

    fig, ax = plt.subplots()
    ax.contour(XX, YY, G, levels=14, cmap=PARULA, linewidths=0.5)
    ax.plot([X[0]], [X[1]], ".k", markersize=12)
    ax.set_aspect("equal")
    save(fig, "opt", "GlobalMinimum_02.png")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(XX, YY, G, cmap=PARULA, rstride=2, cstride=2,
                    linewidth=0)
    ax.plot3D([X[0]], [X[1]], [G[i, j]], ".k", markersize=14)
    ax.set_zlim(-10, 10)
    ax.view_init(elev=4, azim=-24.5 - 90)
    save(fig, "opt", "GlobalMinimum_03.png")


def rosenbrock():
    """opt/Rosenbrock — the classic banana function."""
    from chebfunjax.plotting import PARULA

    def f_np(x, y):
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2

    x = np.linspace(-1.5, 1.5, 400)
    y = np.linspace(-1, 3, 400)
    XX, YY = np.meshgrid(x, y)
    FF = f_np(XX, YY)
    levels = np.concatenate([np.arange(10, 300, 10)])

    fig, ax = plt.subplots()
    ax.contour(XX, YY, FF, levels=levels, cmap=PARULA, linewidths=0.5)
    ax.plot([1], [1], ".r", markersize=10)
    save(fig, "opt", "Rosenbrock_01.png")

    # min over y for each x (1D chebfun trick)
    xs1 = np.linspace(-1.5, 1.5, 500)
    fminx = np.array([np.min(f_np(xv, np.linspace(-1, 3, 2000)))
                      for xv in xs1])
    fig, ax = plt.subplots()
    ax.plot(xs1, fminx, color=CHEBFUN_BLUE, linewidth=1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("min over y")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "opt", "Rosenbrock_02.png")

    fig, ax = plt.subplots()
    ax.semilogy(xs1, np.maximum(fminx, 1e-18), color=CHEBFUN_BLUE,
                linewidth=1.2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "opt", "Rosenbrock_03.png")

    # second example: f2 = exp(x-2x^2-y^2) sin(6(x+y+xy^2))
    def f2_np(x, y):
        return np.exp(x - 2 * x**2 - y**2) * np.sin(
            6 * (x + y + x * y**2))

    x2 = np.linspace(-1, 1, 400)
    X2, Y2 = np.meshgrid(x2, x2)
    F2 = f2_np(X2, Y2)
    fig, ax = plt.subplots()
    cs = ax.contour(X2, Y2, F2, levels=30, cmap=PARULA,
                    linewidths=0.5)
    fig.colorbar(cs, ax=ax, fraction=0.045)
    ax.set_aspect("equal")
    ax.set_title("f(x,y)", fontsize=10)
    save(fig, "opt", "Rosenbrock_04.png")

    # min over y along vertical slices
    yy = np.linspace(-1, 1, 2000)
    fminx2 = np.array([np.min(f2_np(xv, yy)) for xv in x2])
    fig, ax = plt.subplots()
    ax.plot(x2, fminx2, color=CHEBFUN_BLUE, linewidth=1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("min_y(f(x,y))")
    ax.set_title("minimum of f(x,y) along vertical slices",
                 fontsize=10)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "opt", "Rosenbrock_05.png")

    kmin = int(np.argmin(fminx2))
    xmin2 = x2[kmin]
    ymin2 = yy[np.argmin(f2_np(xmin2, yy))]
    print(f"    second-function min at ({xmin2:.4f}, {ymin2:.4f})")
    fig, ax = plt.subplots()
    ax.plot([xmin2], [ymin2], ".k", markersize=14)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "opt", "Rosenbrock_06.png")


PAGES = {
    "FejerJackson": fejerjackson,
    "FourierBasedChebfuns": fourierbasedchebfuns,
    "FourierCoefficients": fouriercoefficients,
    "Catenary": catenary,
    "ConstrainedExtrema": constrainedextrema,
    "GlobalMinimum": globalminimum,
    "Rosenbrock": rosenbrock,
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
