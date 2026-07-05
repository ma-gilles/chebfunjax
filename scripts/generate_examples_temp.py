"""Generate per-block figures for the docs/examples/temp pages."""

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

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images", "temp")
REF = ("/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/refs/"
       "docs/images/temp")
ORANGE = "#D95319"
PI = float(np.pi)


def save(fig, name):
    from PIL import Image

    ref_path = os.path.join(REF, name)
    size = Image.open(ref_path).size if os.path.exists(ref_path) else (600, 270)
    save_chebfun_figure(fig, os.path.join(OUT, name), size=size)
    plt.close(fig)
    print(f"  {name} saved")


def taylorstheorem():
    """temp/TaylorsTheorem — local approximations and convergence disks."""
    # Fig 1: sin with local Chebyshev approximations of growing width
    f = cj.chebfun(lambda x: jnp.sin(x), domain=[-7.0, 7.0])
    xs = jnp.linspace(-7, 7, 1200)
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), np.asarray(f(xs)), color=CHEBFUN_BLUE,
            linewidth=1.2)
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k in range(2, 9):
        w = k * PI / 8
        p = cj.chebfun(lambda x: jnp.sin(x), domain=[-w, w], n=6)
        xw = jnp.linspace(-w, w, 300)
        ax.plot(np.asarray(xw), np.asarray(p(xw)),
                color=cyc[(k - 2) % len(cyc)], linewidth=1.0)
    ax.set_ylim(-2, 2)
    save(fig, "TaylorsTheorem_01.png")

    # Figs 2-5: log|x - i| and approximations on windows around x0=2
    def func_np(x):
        return np.log(np.abs(x - 1j))

    fl = cj.chebfun(lambda x: jnp.log(jnp.abs(x - 1j)), domain=[-5.0, 5.0])
    xs5 = jnp.linspace(-5, 5, 1200)

    def base_plot():
        fig, ax = plt.subplots()
        ax.plot(np.asarray(xs5), np.asarray(fl(xs5)), "k-", linewidth=1.2)
        return fig, ax

    fig, ax = base_plot()
    save(fig, "TaylorsTheorem_02.png")

    x0 = 2.0
    for k, r in enumerate((0.5, 1.2), 3):
        fig, ax = base_plot()
        pr = cj.chebfun(lambda x: jnp.log(jnp.abs(x - 1j)),
                        domain=[x0 - r, x0 + r])
        xw = jnp.linspace(x0 - r, x0 + r, 400)
        ax.plot(np.asarray(xw), np.asarray(pr(xw)), color=ORANGE,
                linewidth=1.6)
        save(fig, f"TaylorsTheorem_{k:02d}.png")

    # Figs 5-7: convergence-disk pictures in the complex plane
    th = np.linspace(0, 2 * PI, 300)
    for k, r in enumerate((0.5, 1.2, np.sqrt(5)), 5):
        fig, ax = plt.subplots()
        ax.plot([-5, 5], [0, 0], "k-", linewidth=1.0)
        ax.plot([x0], [0], ".k", markersize=8)
        ax.plot([0], [1], "xr", markersize=10, markeredgewidth=2)
        ax.plot(x0 + r * np.cos(th), r * np.sin(th), color=CHEBFUN_BLUE,
                linewidth=1.2)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.4, linewidth=0.4)
        save(fig, f"TaylorsTheorem_{k:02d}.png")


def _rl_frac_int(f_np, alpha, xg, a=0.0):
    """Riemann-Liouville fractional integral J^alpha f on grid xg.

    Direct quadrature of 1/Gamma(a) int_a^x (x-t)^(alpha-1) f(t) dt with
    the endpoint singularity handled by substitution t = x - u^(1/alpha).
    """
    from math import gamma

    out = np.zeros_like(xg)
    for i, x in enumerate(xg):
        if x <= a:
            out[i] = 0.0
            continue
        # substitution u = (x - t)^alpha: dt = -(1/alpha) u^(1/alpha - 1) du
        u = np.linspace(0.0, (x - a) ** alpha, 600)[1:]
        t = x - u ** (1.0 / alpha)
        vals = f_np(t)
        integrand = vals / alpha  # (x-t)^{alpha-1} dt = du/alpha
        out[i] = np.trapezoid(integrand, u) / gamma(alpha)
    return out


def fraccalc():
    """temp/FracCalc — fractional derivatives via RL quadrature."""
    from math import gamma

    xg = np.linspace(1e-9, 4.0, 300)

    fig, ax = plt.subplots()
    ax.plot(xg, xg, linewidth=1.2)
    ax.plot(xg, np.ones_like(xg), linewidth=1.2)
    ax.plot(xg, xg**2 / 2, linewidth=1.2)
    ax.set_xlim(0, 4)
    save(fig, "FracCalc_01.png")

    # half-derivative of x: d^{1/2} x = 2 sqrt(x/pi)
    ax_lim = (0, 4, 0, 4)
    fig, ax = plt.subplots()
    ax.plot(xg, xg, linewidth=1.2)
    ax.plot(xg, np.ones_like(xg), linewidth=1.2)
    ax.plot(xg, xg**2 / 2, linewidth=1.2)
    ax.plot(xg, 2 * np.sqrt(xg / PI), linewidth=1.4)
    ax.set_xlim(ax_lim[0], ax_lim[1])
    ax.set_ylim(ax_lim[2], ax_lim[3])
    save(fig, "FracCalc_02.png")

    # family d^alpha x for alpha = 0.1..1 via closed form
    fig, ax = plt.subplots()
    for alpha in np.arange(0.1, 1.01, 0.1):
        da = xg ** (1 - alpha) / gamma(2 - alpha)
        ax.plot(xg, da, linewidth=1.0)
    ax.plot(xg, xg, linewidth=1.2)
    ax.set_xlim(0, 4)
    save(fig, "FracCalc_03.png")

    # fractional derivatives of sin on [0, 20] via RL of the derivative
    xg20 = np.linspace(1e-6, 20.0, 400)
    fig, ax = plt.subplots()
    for alpha in np.sqrt(2) * np.arange(0, 11, 2) / 17:
        if alpha == 0:
            ax.plot(xg20, np.sin(xg20), linewidth=0.9)
        else:
            d = _rl_frac_int(np.cos, 1 - alpha, xg20)
            ax.plot(xg20, d, linewidth=0.9)
    save(fig, "FracCalc_04.png")

    # remaining panels: half-integrals of exp and the family for sin
    fig, ax = plt.subplots()
    J = _rl_frac_int(np.exp, 0.5, xg)
    ax.plot(xg, J, "b", linewidth=1.3)
    from scipy.special import erf

    exact = np.exp(xg) * erf(np.sqrt(xg))
    ax.plot(xg, exact, "--r", linewidth=1.3)
    ax.set_title("Half-integral of exp(x)", fontsize=9)
    save(fig, "FracCalc_05.png")

    fig, ax = plt.subplots()
    for alpha in np.arange(0.1, 1.01, 0.1):
        Jd = _rl_frac_int(np.sin, alpha, xg20)
        ax.plot(xg20, Jd, linewidth=0.9)
    save(fig, "FracCalc_06.png")

    fig, ax = plt.subplots()
    err = np.abs(J - exact)
    ax.semilogy(xg, np.maximum(err, 1e-18), linewidth=1.2)
    ax.set_title("Quadrature error of the half-integral", fontsize=9)
    save(fig, "FracCalc_07.png")


def fraccalc2():
    """temp/FracCalc2 — fractional integrals of Legendre/Jacobi and exp."""
    from math import gamma

    from numpy.polynomial import chebyshev as npcheb
    from numpy.polynomial import legendre as npleg

    xg = np.linspace(-1 + 1e-9, 1.0, 400)

    def P(n):
        c = np.zeros(n + 1)
        c[n] = 1.0
        return lambda t: npleg.legval(t, c)

    def T(n):
        c = np.zeros(n + 1)
        c[n] = 1.0
        return lambda t: npcheb.chebval(t, c)

    n = 4
    J1 = _rl_frac_int(P(n), 0.5, xg, a=-1.0)
    J2 = (T(n)(xg) + T(n + 1)(xg)) / (gamma(0.5) * (n + 0.5)
                                      * np.sqrt(1 + xg))
    fig, ax = plt.subplots()
    ax.plot(xg, J1, "b", linewidth=1.3)
    ax.plot(xg, J2, "--r", linewidth=1.3)
    ax.set_title("Half-integral of P_4 (quadrature vs closed form)",
                 fontsize=9)
    save(fig, "FracCalc2_01.png")

    fig, ax = plt.subplots()
    ax.semilogy(xg, np.maximum(np.abs(J1 - J2), 1e-18), linewidth=1.2)
    save(fig, "FracCalc2_02.png")

    Je = _rl_frac_int(np.exp, 0.5, xg, a=-1.0)
    from scipy.special import erf

    exacte = np.exp(xg) * erf(np.sqrt(xg + 1))
    fig, ax = plt.subplots()
    ax.plot(xg, Je, "b", linewidth=1.3)
    ax.plot(xg, exacte, "--r", linewidth=1.3)
    ax.set_title("Half-integral of exp(x)", fontsize=9)
    save(fig, "FracCalc2_03.png")

    Jq = _rl_frac_int(np.exp, 0.25, xg, a=-1.0)
    fig, ax = plt.subplots()
    ax.plot(xg, Jq, "b", linewidth=1.3)
    ax.set_title("Quarter-integral of exp(x)", fontsize=9)
    save(fig, "FracCalc2_04.png")

    fig, ax = plt.subplots()
    for mu in (0.1, 0.25, 0.5, 0.75, 1.0):
        ax.plot(xg, _rl_frac_int(np.exp, mu, xg, a=-1.0), linewidth=1.0)
    save(fig, "FracCalc2_05.png")

    fig, ax = plt.subplots()
    ax.semilogy(xg, np.maximum(np.abs(Je - exacte), 1e-18), linewidth=1.2)
    save(fig, "FracCalc2_06.png")


def binousshaikhbellagi():
    """temp/BinousShaikhBellagi — transport BVPs via Chebop."""
    import warnings

    from chebfunjax.operators.chebop import Chebop

    # Fig 1-2: 1D convection-diffusion steady states
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L = Chebop(lambda x, u: 0.1 * u.diff(2) - u.diff(),
                   domain=(0.0, 1.0), lbc=1.0, rbc=2.0)
        u = L.solve(0.0)
    xs = jnp.linspace(0, 1, 500)
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), np.asarray(u(xs)), color=CHEBFUN_BLUE,
            linewidth=1.4)
    ax.set_ylim(0, 2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "BinousShaikhBellagi_01.png")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L1 = Chebop(lambda x, u: u.diff(2) + 3 * u.diff() + 2 * u,
                    domain=(0.0, 1.0), lbc=1.0, rbc=0.0)
        u1 = L1.solve(0.0)
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), np.asarray(u1(xs)), color=CHEBFUN_BLUE,
            linewidth=1.4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "BinousShaikhBellagi_02.png")

    # Fig 3: convection-diffusion on [0, 10]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L2 = Chebop(lambda x, c: 0.49 * c.diff(2) - 2.5 * c.diff(),
                    domain=(0.0, 10.0), lbc=1.0, rbc=0.0)
        c = L2.solve(0.0)
    xs10 = jnp.linspace(0, 10, 600)
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs10), np.asarray(c(xs10)), color=CHEBFUN_BLUE,
            linewidth=1.4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "BinousShaikhBellagi_03.png")

    # Fig 4: Falkner-Skan-type nonlinear third-order BVP on [0, 4]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        N = Chebop(
            lambda x, f: f.diff(3) + f * f.diff(2)
            + (PI / 4) * (1 - f.diff() ** 2),
            domain=(0.0, 4.0))
        N.lbc = [0.0, 0.0]
        N.rbc = 1.0
        try:
            fbl = N.solve(0.0)
            xs4 = jnp.linspace(0, 4, 400)
            vals = np.asarray(fbl(xs4))
        except Exception:
            # boundary-layer fallback: shoot with solve_ivp on f''' = ...
            from scipy.integrate import solve_ivp
            from scipy.optimize import brentq

            def rhs(t, y):
                return [y[1], y[2],
                        -y[0] * y[2] - (PI / 4) * (1 - y[1] ** 2)]

            def shoot(s):
                sol = solve_ivp(rhs, [0, 4], [0, 0, s], rtol=1e-10)
                return sol.y[1][-1] - 1.0

            s0 = brentq(shoot, 0.1, 3.0)
            sol = solve_ivp(rhs, [0, 4], [0, 0, s0], rtol=1e-10,
                            t_eval=np.linspace(0, 4, 400))
            xs4 = sol.t
            vals = sol.y[0]
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs4), vals, color=CHEBFUN_BLUE, linewidth=1.4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "BinousShaikhBellagi_04.png")

    # Figs 5-6: derivative profile and a heat-equation snapshot via expm
    fig, ax = plt.subplots()
    dv = np.gradient(np.asarray(vals).ravel(),
                     np.asarray(xs4).ravel())
    ax.plot(np.asarray(xs4), dv, color=CHEBFUN_BLUE, linewidth=1.4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "BinousShaikhBellagi_05.png")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A = Chebop(lambda x, u: u.diff(2), domain=(0.0, 1.0),
                   lbc=0.0, rbc=0.0)
        u0 = cj.chebfun(lambda x: jnp.sin(jnp.pi * x), domain=[0.0, 1.0])
        fig, ax = plt.subplots()
        for t in (0.0, 0.02, 0.05, 0.1, 0.2):
            ut = u0 if t == 0 else A.expm(t, u0, n=64)
            ax.plot(np.asarray(xs), np.asarray(ut(xs)), linewidth=1.2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "BinousShaikhBellagi_06.png")


PAGES = {
    "TaylorsTheorem": taylorstheorem,
    "FracCalc2": fraccalc2,
    "FracCalc": fraccalc,
    "BinousShaikhBellagi": binousshaikhbellagi,
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
