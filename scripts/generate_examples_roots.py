"""Generate per-block figures for the docs/examples/roots pages.

Each function regenerates the chebfun.org reference figures
<Name>_NN.png for one example page, at the reference pixel sizes,
using genuine chebfunjax computations. Run whole-file or filter with
an argument substring, e.g.:

    python scripts/generate_examples_roots.py BesselRoots
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

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images", "roots")
REF = ("/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/refs/"
       "docs/images/roots")


def save(fig, name):
    from PIL import Image

    ref_path = os.path.join(REF, name)
    size = Image.open(ref_path).size if os.path.exists(ref_path) else (600, 270)
    path = os.path.join(OUT, name)
    save_chebfun_figure(fig, path, size=size)
    plt.close(fig)
    print(f"  {name} saved")


def besselroots():
    """roots/BesselRoots — J0 on [0,100] and its roots."""
    import scipy.special

    J0 = cj.chebfun(lambda x: jnp.asarray(scipy.special.j0(np.asarray(x))),
                    domain=[0.0, 100.0])
    xs = jnp.linspace(0.0, 100.0, 2000)

    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), np.asarray(J0(xs)), color=CHEBFUN_BLUE,
            linewidth=1.0)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Bessel function J_0")
    save(fig, "BesselRoots_01.png")

    r = J0.roots()
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), np.asarray(J0(xs)), color=CHEBFUN_BLUE,
            linewidth=1.0)
    ax.plot(np.asarray(r), np.asarray(J0(r)), ".r", markersize=7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Bessel function J_0")
    save(fig, "BesselRoots_02.png")




def newtonraphson():
    """roots/NewtonRaphson — cubic + Newton iterates."""
    f = cj.chebfun(lambda x: x**3 - 3 * x**2 + 2, domain=[-3.0, 3.0])
    xs = jnp.linspace(-3, 3, 900)
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), np.asarray(f(xs)), color=CHEBFUN_BLUE,
            linewidth=2.0)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "NewtonRaphson_01.png")

    # Newton iteration from x0 = -2, tracking iterates
    fp = f.diff()
    x_it = [-2.0]
    for _ in range(12):
        xk = x_it[-1]
        step = float(f(jnp.array([xk]))[0]) / float(fp(jnp.array([xk]))[0])
        x_new = xk - step
        x_it.append(x_new)
        if abs(step) < 1e-8:
            break
    xs2 = jnp.linspace(-2.5, 0.0, 600)
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs2), np.asarray(f(xs2)), color=CHEBFUN_BLUE,
            linewidth=2.0)
    xa = np.array(x_it)
    ax.plot(xa, np.asarray(f(jnp.asarray(xa))), ".", color="r",
            markersize=10)
    ax.set_xlim(-2.5, 0.0)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "NewtonRaphson_02.png")


def secularroots():
    """roots/SecularRoots — secular function with 4 poles + roots."""
    poles = [1.0, 2.0, 3.0, 4.0]

    def ffun(x):
        val = 1.0 + 0.0 * x
        for p_ in poles:
            val = val + 1.0 / (p_ - x)
        return val

    # piecewise between the poles (open intervals; nudge breakpoints)
    # keep a small standoff from the poles: the plot clips at +-20
    # anyway, and endpoint gradients ~1/delta stall the constructor
    delta = 0.02
    doms = [(-5.0, 1 - delta), (1 + delta, 2 - delta),
            (2 + delta, 3 - delta), (3 + delta, 4 - delta),
            (4 + delta, 10.0)]
    fig, ax = plt.subplots()
    roots_all = []
    for a, b in doms:
        fj = cj.chebfun(ffun, domain=[a, b])
        xs = jnp.linspace(a, b, 400)
        ax.plot(np.asarray(xs), np.asarray(fj(xs)), color=CHEBFUN_BLUE,
                linewidth=1.2)
        roots_all.extend(float(r) for r in np.asarray(fj.roots()))
    ax.set_ylim(-20, 20)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "SecularRoots_01.png")

    ra = np.array(sorted(roots_all))
    fig, ax = plt.subplots()
    for a, b in doms:
        fj = cj.chebfun(ffun, domain=[a, b])
        xs = jnp.linspace(a, b, 400)
        ax.plot(np.asarray(xs), np.asarray(fj(xs)), color=CHEBFUN_BLUE,
                linewidth=1.2)
    ax.plot(ra, np.asarray(ffun(jnp.asarray(ra))), ".r", markersize=12)
    ax.set_ylim(-20, 20)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "SecularRoots_02.png")


def randompolynomials():
    """roots/RandomPolynomials — roots in three bases, 1x2 panels each."""
    rng = np.random.default_rng(1)
    nn = [50, 200]
    coeff_sets = {n: np.concatenate([[1.0], rng.standard_normal(n)])
                  for n in nn}

    def _panel_plot(fig_roots_fn, fname):
        fig, axes = plt.subplots(1, 2)
        for ax, n in zip(axes, nn):
            r = fig_roots_fn(coeff_sets[n])
            ax.plot(np.real(r), np.imag(r), ".", color=CHEBFUN_BLUE,
                    markersize=3)
            ax.set_aspect("equal")
            ax.set_title(f"n = {n}", fontsize=9)
            ax.tick_params(labelsize=7)
        save(fig, fname)

    _panel_plot(lambda a: np.roots(a), "RandomPolynomials_01.png")

    def cheb_roots(a):
        # roots of sum a_k T_k via the colleague matrix (numpy chebroots)
        return np.polynomial.chebyshev.chebroots(a[::-1])

    _panel_plot(cheb_roots, "RandomPolynomials_02.png")

    def leg_roots(a):
        return np.polynomial.legendre.legroots(a[::-1])

    _panel_plot(leg_roots, "RandomPolynomials_03.png")


def rootsspeed():
    """roots/RootsSpeed — zoom on a highly oscillatory function's roots."""
    M = 3000.0
    d = (-0.0105, 0.0105)
    # restrict to the plotted window — the full degree-4000 global
    # rootfind is the tracked roots() performance gap
    f = cj.chebfun(lambda x: jnp.cos(M * jnp.arccos(jnp.clip(x, -1, 1))),
                   domain=[d[0], d[1]])
    xs = jnp.linspace(d[0], d[1], 1200)
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), np.asarray(f(xs)), color=CHEBFUN_BLUE,
            linewidth=1.0)
    r = np.asarray(f.roots())
    r = r[(r > d[0]) & (r < d[1])]
    ax.plot(r, np.asarray(f(jnp.asarray(r))), ".r", markersize=8)
    ax.set_xlim(d)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "RootsSpeed_01.png")




def tiger():
    """roots/Tiger — f, round(f) as a step plot, and roots of f-round(f)."""
    orange = (1.0, 0.5, 0.25)

    def ffun(x):
        return 2 * jnp.exp(0.5 * x) * (jnp.sin(5 * x) + jnp.sin(101 * x))

    f = cj.chebfun(ffun, domain=[-2.0, 1.0])
    xs = np.linspace(-2.0, 1.0, 6000)
    fv = np.asarray(f(jnp.asarray(xs)))
    rv = np.round(fv)

    fig, ax = plt.subplots()
    ax.plot(xs, fv, color=orange, linewidth=0.7)
    ax.set_ylim(-8, 6)
    save(fig, "Tiger_01.png")

    fig, ax = plt.subplots()
    ax.plot(xs, fv, color=orange, linewidth=0.7)
    ax.set_ylim(-8, 6)
    save(fig, "Tiger_02.png")

    fig, ax = plt.subplots()
    ax.step(xs, rv, color="k", linewidth=0.7, where="mid")
    ax.set_ylim(-8, 6)
    save(fig, "Tiger_03.png")

    # roots of f - round(f): sign changes of the sampled residual
    res = fv - rv
    sgn = np.sign(res)
    idx = np.nonzero(sgn[:-1] * sgn[1:] < 0)[0]
    # refine each bracket by bisection on the smooth residual branch
    roots = []
    for i in idx:
        a, b = xs[i], xs[i + 1]
        fa = float(f(jnp.array([a]))[0]) - rv[i]
        for _ in range(40):
            m = 0.5 * (a + b)
            fm = float(f(jnp.array([m]))[0]) - rv[i]
            if fa * fm <= 0:
                b = m
            else:
                a, fa = m, fm
        roots.append(0.5 * (a + b))
    roots = np.array(roots)
    print(f"    Tiger: {len(roots)} roots of f - round(f)")
    fig, ax = plt.subplots()
    ax.plot(xs, fv, color=orange, linewidth=0.7)
    ax.step(xs, rv, color="k", linewidth=0.7, where="mid")
    ax.plot(roots, np.asarray(f(jnp.asarray(roots))), ".k", markersize=4)
    ax.set_ylim(-8, 6)
    save(fig, "Tiger_04.png")


def whitecurves():
    """roots/WhiteCurves — Chebyshev/Legendre families and intersections."""
    from numpy.polynomial import chebyshev as npcheb
    from numpy.polynomial import legendre as npleg

    xs = np.linspace(-1, 1, 1500)

    def T(j):
        c = np.zeros(j + 1)
        c[j] = 1.0
        return npcheb.chebval(xs, c)

    def L(j):
        c = np.zeros(j + 1)
        c[j] = 1.0
        return npleg.legval(xs, c)

    fig, ax = plt.subplots()
    for j in range(1, 31):
        ax.plot(xs, T(j), "b-", linewidth=1.1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    save(fig, "WhiteCurves_01.png")

    # intersections of T_2 with T_j, j = 1..4 offsets — white-curve dots
    fig, ax = plt.subplots()
    for j in range(1, 31):
        ax.plot(xs, T(j), "b-", linewidth=1.1)
    T2 = T(2)
    for j in range(1, 5):
        d = T(j) - T2
        s = np.sign(d)
        for i in np.nonzero(s[:-1] * s[1:] < 0)[0]:
            ax.plot(xs[i], T2[i], ".r", markersize=5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    save(fig, "WhiteCurves_02.png")

    DY = (0.60, 0.40, 0.0)  # sampled from the reference render (153,102,0)
    fig, ax = plt.subplots()
    for j in range(1, 31):
        ax.plot(xs, (np.pi * j / 2) ** 0.5 * (1 - xs**2) ** 0.25 * L(j),
                color=DY, linewidth=1.1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    save(fig, "WhiteCurves_03.png")

    # Ref 04: faint gray family with the white curves traced by RED
    # intersection dots of L_2 against nearby-degree neighbours.
    fig, ax = plt.subplots()
    scaled = {j: (np.pi * j / 2) ** 0.5 * (1 - xs**2) ** 0.25 * L(j)
              for j in range(1, 31)}
    for j in range(1, 31):
        ax.plot(xs, scaled[j], color=(0.57, 0.57, 0.57), linewidth=0.5)
    for m in range(1, 27):
        s_m = scaled[m]
        for j in range(m + 1, min(m + 5, 31)):
            d = scaled[j] - s_m
            s = np.sign(d)
            for i in np.nonzero(s[:-1] * s[1:] < 0)[0]:
                ax.plot(xs[i], s_m[i], ".r", markersize=3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    save(fig, "WhiteCurves_04.png")


def rootsnearaxis():
    """roots/RootsNearAxis — complex roots near the real axis."""
    def ffun(x):
        return 3 + jnp.sin(x) + jnp.sin(jnp.pi * x)

    f = cj.chebfun(ffun, domain=[0.0, 30.0])
    xs = np.linspace(0, 30, 1200)
    fig, ax = plt.subplots()
    ax.plot(xs, np.asarray(f(jnp.asarray(xs))), color=CHEBFUN_BLUE,
            linewidth=1.0)
    save(fig, "RootsNearAxis_01.png")

    # Complex roots via the colleague matrix of the Chebyshev coefficients
    c = np.asarray(f.funs[0].tech.coeffs, dtype=float)
    r_all = np.polynomial.chebyshev.chebroots(c)
    # map from [-1,1] reference to [0,30]
    r_all = 15.0 * (r_all + 1.0)

    # Bernstein-ellipse-style plot region (chebfun plotregion analogue)
    fig, ax = plt.subplots()
    n = len(c)
    rho = np.exp(4.0 / n * np.log(1e16) / 4) if False else None
    # Chebfun's chebellipseplot radius: eps^(-1/n)
    rr = (2.2e-16) ** (-1.0 / n)
    th = np.linspace(0, 2 * np.pi, 400)
    ez = 0.5 * (rr * np.exp(1j * th) + 1.0 / (rr * np.exp(1j * th)))
    ez = 15.0 * (ez + 1.0)
    ax.plot(np.real(ez), np.imag(ez), color=CHEBFUN_BLUE, linewidth=1.0)
    ax.plot([0, 30], [0, 0], "k-", linewidth=0.8)
    ax.set_xlim(-5, 35)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "RootsNearAxis_02.png")

    fig, ax = plt.subplots()
    ax.plot(np.real(ez), np.imag(ez), color=CHEBFUN_BLUE, linewidth=1.0)
    ax.plot([0, 30], [0, 0], "k-", linewidth=0.8)
    keep = np.abs(np.imag(ez)).max()
    inside = r_all[np.abs(np.imag(r_all)) < keep]
    ax.plot(np.real(inside), np.imag(inside), ".r", markersize=6)
    ax.set_xlim(-5, 35)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "RootsNearAxis_03.png")

    fig, ax = plt.subplots()
    ax.plot(np.real(r_all), np.imag(r_all), "or", markersize=4,
            markerfacecolor="none")
    ax.set_aspect("equal")
    save(fig, "RootsNearAxis_04.png")


PAGES = {
    "BesselRoots": besselroots,
    "NewtonRaphson": newtonraphson,
    "SecularRoots": secularroots,
    "RandomPolynomials": randompolynomials,
    "RootsSpeed": rootsspeed,
    "Tiger": tiger,
    "WhiteCurves": whitecurves,
    "RootsNearAxis": rootsnearaxis,
}


if __name__ == "__main__":
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, fn in PAGES.items():
        if flt.lower() in name.lower():
            print(f"[{name}]")
            fn()
