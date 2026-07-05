"""Generate per-block figures for the stats example category, tranche 1:
CentralLimitTheorem, Smoothies, ResamplingRandomVariables, Expectations,
ProbabilityConvolution.
"""

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


def save(fig, name):
    from PIL import Image

    ref_path = os.path.join(REFROOT, "stats", name)
    size = Image.open(ref_path).size if os.path.exists(ref_path) else (600, 270)
    save_chebfun_figure(fig, os.path.join(DOCS, "stats", name),
                        size=size)
    plt.close(fig)
    print(f"  stats/{name} saved")


def _conv_grid(f_vals, g_vals, xs):
    """Numerical convolution of two densities sampled on the same
    uniform grid xs; returns values on xs (centered support)."""
    dx = xs[1] - xs[0]
    full = np.convolve(f_vals, g_vals) * dx
    n = len(xs)
    mid_start = (len(full) - n) // 2
    return full[mid_start:mid_start + n]


def centrallimittheorem():
    """stats/CentralLimitTheorem — convolving a skew density."""
    # X: triangular-ish density (4/3+x)/2 on [-4/3, 2/3]
    def X_pdf(x):
        x = np.asarray(x)
        return np.where((x >= -4 / 3) & (x <= 2 / 3), (4 / 3 + x) / 2,
                        0.0)

    xs = np.linspace(-3, 3, 2401)
    ax_lim = (-3, 3, -0.2, 1.2)

    fig, ax = plt.subplots()
    ax.plot(xs, X_pdf(xs), color=CHEBFUN_BLUE, linewidth=1.2)
    ax.set_xlim(ax_lim[0], ax_lim[1])
    ax.set_ylim(ax_lim[2], ax_lim[3])
    save(fig, "CentralLimitTheorem_01.png")

    mean = np.trapezoid(xs * X_pdf(xs), xs)
    variance = np.trapezoid((xs - mean) ** 2 * X_pdf(xs), xs)
    sigma = np.sqrt(variance)
    print(f"    mean {mean:.4f}, variance {variance:.4f}")

    def gauss(sig, x):
        return np.exp(-0.5 * (np.asarray(x) / sig) ** 2) / (
            sig * np.sqrt(2 * PI))

    fig, ax = plt.subplots()
    ax.plot(xs, X_pdf(xs), color=CHEBFUN_BLUE, linewidth=1.2)
    ax.plot(xs, gauss(sigma, xs - mean), "r", linewidth=1.2)
    ax.set_xlim(ax_lim[0], ax_lim[1])
    ax.set_ylim(ax_lim[2], ax_lim[3])
    ax.set_title("X and the matching Gaussian", fontsize=10)
    save(fig, "CentralLimitTheorem_02.png")

    # iterated self-convolutions of the centered density
    Xc = X_pdf(xs + mean)  # centered
    Sk = Xc.copy()
    panel = 3
    for k in range(2, 7):
        Sk = _conv_grid(Sk, Xc, xs)
        # normalized sum density: S_k has variance k sigma^2; rescale
        # to unit-comparable axes like the example's plots
        if panel <= 8:
            fig, ax = plt.subplots()
            ax.plot(xs, Sk, color=CHEBFUN_BLUE, linewidth=1.2)
            ax.plot(xs, gauss(sigma * np.sqrt(k), xs), "r",
                    linewidth=1.0)
            ax.set_xlim(-3, 3)
            ax.set_title(f"sum of {k} copies vs Gaussian", fontsize=10)
            save(fig, f"CentralLimitTheorem_{panel:02d}.png")
            panel += 1

    # error decay vs k
    errs = []
    Sk = Xc.copy()
    ks = range(2, 12)
    for k in ks:
        Sk = _conv_grid(Sk, Xc, xs)
        errs.append(np.max(np.abs(Sk - gauss(sigma * np.sqrt(k), xs))))
    fig, ax = plt.subplots()
    ax.semilogy(list(ks), errs, ".-", color=CHEBFUN_BLUE,
                markersize=7, linewidth=0.9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_xlabel("number of summands k")
    ax.set_ylabel("max deviation from Gaussian")
    save(fig, "CentralLimitTheorem_08.png")


def smoothies():
    """stats/Smoothies — random C-infinity functions."""
    rng = np.random.default_rng(1)

    def smoothie(seed, m=30, complex_=False):
        r = np.random.default_rng(seed)
        # random coefficients with Gaussian decay: C-infinity texture
        k = np.arange(m + 1)
        decay = np.exp(-((k / (m / 3.0)) ** 2))
        a = r.standard_normal(m + 1) * decay
        b = r.standard_normal(m + 1) * decay
        if complex_:
            c = (r.standard_normal(m + 1)
                 + 1j * r.standard_normal(m + 1)) * decay

        def f(x):
            s = PI * (np.asarray(x) + 1)
            out = sum(a[j] * np.cos(j * s) + b[j] * np.sin(j * s)
                      for j in range(m + 1))
            return out * 1.2

        def fc(x):
            s = PI * (np.asarray(x) + 1)
            return sum(c[j] * np.exp(1j * j * s)
                       for j in range(m + 1))

        return (fc if complex_ else f,
                (np.abs(np.concatenate([a, b])) if not complex_
                 else np.abs(c)))

    xs = np.linspace(-1, 1, 2000)
    f, coeffs = smoothie(1)
    fig, ax = plt.subplots()
    ax.plot(xs, f(xs), color=CHEBFUN_BLUE, linewidth=1.0)
    ax.set_ylim(-4, 4)
    save(fig, "Smoothies_01.png")

    fig, ax = plt.subplots()
    ax.semilogy(np.arange(len(coeffs)), np.maximum(coeffs, 1e-18),
                ".k", markersize=4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("coefficients: Gaussian decay", fontsize=10)
    save(fig, "Smoothies_02.png")

    f2, _ = smoothie(2)
    fig, ax = plt.subplots()
    ax.plot(xs, f2(xs), color=ORANGE, linewidth=1.0)
    ax.set_ylim(-4, 4)
    save(fig, "Smoothies_03.png")

    # complex smoothie in the plane
    fc, _ = smoothie(3, complex_=True)
    zz = fc(xs)
    fig, ax = plt.subplots()
    ax.plot(np.real(zz), np.imag(zz), color=CHEBFUN_BLUE,
            linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_title("a complex smoothie", fontsize=10)
    save(fig, "Smoothies_04.png")

    # periodic smoothie
    f4, c4 = smoothie(4)
    fig, ax = plt.subplots()
    ax.plot(xs, f4(xs), "m", linewidth=1.0)
    ax.set_ylim(-4, 4)
    ax.set_title("a periodic smoothie", fontsize=10)
    save(fig, "Smoothies_05.png")

    fig, ax = plt.subplots()
    ax.semilogy(np.arange(len(c4)), np.maximum(c4, 1e-18), ".k",
                markersize=4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "Smoothies_06.png")


def resamplingrandomvariables():
    """stats/ResamplingRandomVariables — inverse-CDF sampling."""
    # von Mises distribution on [-pi, pi]
    from scipy.special import iv

    kappa = 1.5
    dom = (-PI, PI)
    xs = np.linspace(*dom, 2000)
    density = np.exp(kappa * np.cos(xs)) / (2 * PI * iv(0, kappa))

    fig, ax = plt.subplots()
    ax.plot(xs, density, color=CHEBFUN_BLUE, linewidth=1.4)
    cdf = np.concatenate([[0], np.cumsum(
        0.5 * (density[1:] + density[:-1]) * np.diff(xs))])
    cdf /= cdf[-1]
    ax.plot(xs, cdf, color=ORANGE, linewidth=1.4)
    ax.set_xlim(-PI, PI)
    ax.set_ylim(0, 1)
    ax.set_title("von Mises distribution", fontsize=10)
    save(fig, "ResamplingRandomVariables_01.png")

    # inverse CDF
    us = np.linspace(1e-6, 1 - 1e-6, 1500)
    cdfinv = np.interp(us, cdf, xs)
    fig, ax = plt.subplots()
    ax.plot(us, cdfinv, color=CHEBFUN_BLUE, linewidth=1.4)
    ax.set_title("Inverse of von Mises distribution", fontsize=10)
    save(fig, "ResamplingRandomVariables_02.png")

    # resampled histogram vs density (MATLAB hist styling)
    rng = np.random.default_rng(0)
    samples = np.interp(rng.random(10000), cdf, xs)
    fig, ax = plt.subplots()
    ax.hist(samples, bins=36, density=True,
            color=(0.2, 0.15, 0.5), edgecolor=(0.95, 0.95, 0.6),
            linewidth=0.5)
    ax.plot(xs, density, "r", linewidth=1.4)
    ax.set_xlim(-PI, PI)
    ax.set_title("Sampled points and the orignal density",
                 fontsize=10)
    save(fig, "ResamplingRandomVariables_03.png")

    # a second distribution: bimodal
    dens2 = np.exp(-8 * (xs - 1) ** 2) + 0.7 * np.exp(
        -8 * (xs + 1.2) ** 2)
    dens2 /= np.trapezoid(dens2, xs)
    cdf2 = np.concatenate([[0], np.cumsum(
        0.5 * (dens2[1:] + dens2[:-1]) * np.diff(xs))])
    cdf2 /= cdf2[-1]

    fig, ax = plt.subplots()
    ax.plot(xs, dens2, color=CHEBFUN_BLUE, linewidth=1.4)
    ax.plot(xs, cdf2, color=ORANGE, linewidth=1.4)
    ax.set_title("a bimodal distribution and its CDF", fontsize=10)
    save(fig, "ResamplingRandomVariables_04.png")

    cdfinv2 = np.interp(us, cdf2, xs)
    fig, ax = plt.subplots()
    ax.plot(us, cdfinv2, color=CHEBFUN_BLUE, linewidth=1.4)
    ax.set_title("inverse CDF: steep at the trough", fontsize=10)
    save(fig, "ResamplingRandomVariables_05.png")

    # Beta(2,2)-style semicircular density on [0, 1]
    x01 = np.linspace(0, 1, 1200)
    dens3 = 6 * x01 * (1 - x01)
    cdf3 = np.concatenate([[0], np.cumsum(
        0.5 * (dens3[1:] + dens3[:-1]) * np.diff(x01))])
    cdf3 /= cdf3[-1]
    samples3 = np.interp(rng.random(10000), cdf3, x01)
    fig, ax = plt.subplots()
    ax.hist(samples3, bins=40, density=True,
            color=(0.2, 0.15, 0.5), edgecolor=(0.95, 0.95, 0.6),
            linewidth=0.5)
    ax.plot(x01, dens3, "r", linewidth=1.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.5)
    ax.set_title("Sampled points and the orignal density",
                 fontsize=10)
    save(fig, "ResamplingRandomVariables_06.png")


def expectations():
    """stats/Expectations — moments of an exponential density."""
    dom = (0.0, 6.0)
    f = cj.chebfun(lambda x: 2 * jnp.exp(-2 * x), domain=list(dom))
    xs = np.linspace(*dom, 1500)
    fv = np.asarray(f(jnp.asarray(xs)))

    fig, ax = plt.subplots()
    ax.plot(xs, fv, color=CHEBFUN_BLUE, linewidth=1.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("density f = 2 exp(-2x)", fontsize=10)
    save(fig, "Expectations_01.png")

    xf = cj.chebfun(lambda x: x * 2 * jnp.exp(-2 * x),
                    domain=list(dom))
    fig, ax = plt.subplots()
    ax.plot(xs, np.asarray(xf(jnp.asarray(xs))), color=CHEBFUN_BLUE,
            linewidth=1.6)
    ax.set_ylim(-0.05, 0.4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title(f"x f(x): E[X] = {float(xf.sum()):.4f}", fontsize=10)
    save(fig, "Expectations_02.png")

    x2f = cj.chebfun(lambda x: x**2 * 2 * jnp.exp(-2 * x),
                     domain=list(dom))
    EX = float(xf.sum())
    EX2 = float(x2f.sum())
    print(f"    E[X] = {EX:.6f}, Var = {EX2 - EX**2:.6f}")
    fig, ax = plt.subplots()
    ax.plot(xs, np.asarray(x2f(jnp.asarray(xs))), color=CHEBFUN_BLUE,
            linewidth=1.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title(f"x^2 f(x): E[X^2] = {EX2:.4f}", fontsize=10)
    save(fig, "Expectations_03.png")

    # g(X) = sin(X): E[g(X)]
    gf = cj.chebfun(lambda x: jnp.sin(x) * 2 * jnp.exp(-2 * x),
                    domain=list(dom))
    fig, ax = plt.subplots()
    ax.plot(xs, np.asarray(gf(jnp.asarray(xs))), color=CHEBFUN_BLUE,
            linewidth=1.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title(f"sin(x) f(x): E[sin X] = {float(gf.sum()):.4f}",
                 fontsize=10)
    save(fig, "Expectations_04.png")

    # CDF and median
    F = f.cumsum()
    Fv = np.asarray(F(jnp.asarray(xs)))
    med = xs[np.searchsorted(Fv, 0.5)]
    fig, ax = plt.subplots()
    ax.plot(xs, Fv, color=CHEBFUN_BLUE, linewidth=1.6)
    ax.plot([med], [0.5], ".r", markersize=12)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title(f"CDF; median = {med:.4f} (exact ln2/2 = "
                 f"{np.log(2)/2:.4f})", fontsize=9)
    save(fig, "Expectations_05.png")

    # moment-generating flavor: E[e^{tX}] over t
    tt = np.linspace(-1, 1.6, 200)
    mgf = [float(cj.chebfun(
        lambda x, _t=t: jnp.exp(_t * x) * 2 * jnp.exp(-2 * x),
        domain=list(dom)).sum()) for t in tt]
    fig, ax = plt.subplots()
    ax.plot(tt, mgf, color=CHEBFUN_BLUE, linewidth=1.4)
    ax.plot(tt, 2 / (2 - tt), "r--", linewidth=0.9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("MGF vs exact 2/(2-t)", fontsize=10)
    save(fig, "Expectations_06.png")


def probabilityconvolution():
    """stats/ProbabilityConvolution — sums of random variables."""
    xs = np.linspace(-5, 5, 4001)

    def gauss(mu, sig, x):
        return np.exp(-0.5 * ((np.asarray(x) - mu) / sig) ** 2) / (
            sig * np.sqrt(2 * PI))

    N1 = gauss(-0.2, 0.3, xs)
    N2 = gauss(0.2, 0.4, xs)
    N3 = _conv_grid(N1, N2, xs)
    fig, ax = plt.subplots()
    ax.plot(xs, N1, "b", linewidth=1.6)
    ax.plot(xs, N2, "r", linewidth=1.6)
    ax.plot(xs, N3, "k", linewidth=1.6)
    ax.set_xlim(-1.5, 1.5)
    save(fig, "ProbabilityConvolution_01.png")

    # gamma distributions: conv(gamma(k1), gamma(k2)) = gamma(k1+k2)
    from math import gamma as G

    def gamma_pdf(x, k, t=1.0):
        x = np.asarray(x)
        out = np.zeros_like(x)
        m = x > 0
        out[m] = x[m] ** (k - 1) * np.exp(-x[m] / t) / (t**k * G(k))
        return out

    xg = np.linspace(0, 10, 4001) - 5  # shift grid for conv helper
    xs5 = np.linspace(-5, 5, 4001)
    g1 = gamma_pdf(xs5 + 2.5, 1.5)
    g2 = gamma_pdf(xs5 + 2.5, 2.0)
    g3 = _conv_grid(g1, g2, xs5)
    fig, ax = plt.subplots()
    xplot = xs5 + 5.0  # back to [0, 10]
    ax.plot(xplot, g1, "b", linewidth=1.4)
    ax.plot(xplot, g2, "r", linewidth=1.4)
    ax.plot(xplot, g3, "k", linewidth=1.4)
    ax.plot(xplot, gamma_pdf(xplot - 5.0 + 5.0 - 5.0 + 5.0
                             if False else xplot, 3.5), "g--",
            linewidth=1.0)
    ax.set_xlim(0, 10)
    ax.set_title("conv(gamma(1.5), gamma(2)) = gamma(3.5)",
                 fontsize=9)
    save(fig, "ProbabilityConvolution_02.png")

    # uniform + uniform = triangular; + again = piecewise quadratic
    U = np.where(np.abs(xs) <= 0.5, 1.0, 0.0)
    T = _conv_grid(U, U, xs)
    Q = _conv_grid(T, U, xs)
    fig, ax = plt.subplots()
    ax.plot(xs, U, "b", linewidth=1.4)
    ax.plot(xs, T, "r", linewidth=1.4)
    ax.plot(xs, Q, "k", linewidth=1.4)
    ax.set_xlim(-2, 2)
    ax.set_title("uniform, triangular, piecewise-quadratic",
                 fontsize=9)
    save(fig, "ProbabilityConvolution_03.png")

    # exponential + exponential
    E1 = np.where(xs > 0, np.exp(-xs), 0.0)
    E2 = _conv_grid(E1, E1, xs)
    fig, ax = plt.subplots()
    ax.plot(xs, E1, "b", linewidth=1.4)
    ax.plot(xs, E2, "k", linewidth=1.4)
    ax.plot(xs, np.where(xs > 0, xs * np.exp(-xs), 0), "g--",
            linewidth=1.0)
    ax.set_xlim(-0.5, 5)
    ax.set_title("conv of exponentials = x exp(-x)", fontsize=9)
    save(fig, "ProbabilityConvolution_04.png")

    # mixed: uniform + Gaussian (smoothing a box)
    M = _conv_grid(U, gauss(0, 0.15, xs), xs)
    fig, ax = plt.subplots()
    ax.plot(xs, U, "b", linewidth=1.4)
    ax.plot(xs, M, "k", linewidth=1.4)
    ax.set_xlim(-1.5, 1.5)
    ax.set_title("uniform smoothed by a Gaussian", fontsize=9)
    save(fig, "ProbabilityConvolution_05.png")


PAGES = {
    "CentralLimitTheorem": centrallimittheorem,
    "Smoothies": smoothies,
    "ResamplingRandomVariables": resamplingrandomvariables,
    "Expectations": expectations,
    "ProbabilityConvolution": probabilityconvolution,
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
