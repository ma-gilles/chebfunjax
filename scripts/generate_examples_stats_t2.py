"""Generate per-block figures for the stats example category, tranche 2:
UniformExercises, MercerKarhunenLoeve, SmoothRandomWalk,
BayesianGradebook, RandomSurf, RandomPolynomials, RandomMaxima,
Histogram, BivariateNormalDistribution, NormalExercises, LeastSquares.
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


def randnfun_np(lam, dom, seed):
    rng = np.random.default_rng(seed)
    a, b = dom
    m = int(2 * (b - a) / lam) + 1
    C = rng.standard_normal((m + 1, 2))

    def f(t):
        s = 2 * PI * (np.asarray(t) - a) / (b - a)
        out = sum(C[k, 0] * np.cos(k * s) + C[k, 1] * np.sin(k * s)
                  for k in range(m + 1))
        return out / np.sqrt(m + 1)

    return f


def uniformexercises():
    """stats/UniformExercises — the original exercise panels."""
    # 1: uniform on [1, 2], area over [a, 2], purple
    xs = np.linspace(1, 2, 400)
    a = 1.4
    fig, ax = plt.subplots()
    m = xs >= a
    ax.fill_between(xs[m], 0, np.ones_like(xs[m]),
                    color=(0.3, 0.2, 0.5))
    ax.plot(xs, np.ones_like(xs), "k", linewidth=2)
    ax.set_xlim(1, 2)
    ax.set_ylim(0, 2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "UniformExercises_01.png")

    # 2: uniform on [a, b] = [-2, 2], area over [a, 0], rust
    a2, b2 = -2.0, 2.0
    xs2 = np.linspace(a2, b2, 400)
    dens = np.full_like(xs2, 1 / (b2 - a2))
    fig, ax = plt.subplots()
    m2 = xs2 <= 0
    ax.fill_between(xs2[m2], 0, dens[m2], color=(0.75, 0.3, 0.2))
    ax.plot(xs2, dens, "k", linewidth=1.6)
    ax.set_xlim(a2, b2)
    ax.set_ylim(0, 0.5)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "UniformExercises_02.png")

    # 3: a quadratic g with red dots at its roots aa
    xs3 = np.linspace(-5, 5, 600)
    g = 0.15 * (xs3 + 2.2) * (xs3 - 3.1)
    fig, ax = plt.subplots()
    ax.plot(xs3, g, linewidth=2, color=CHEBFUN_BLUE)
    ax.plot([-5, 5], [0, 0], "-k", linewidth=0.8)
    ax.plot([-2.2, 3.1], [0, 0], "r.", markersize=14)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "UniformExercises_03.png")

    # 4: the color-wheel spinner: uniform on [0, 360] with bands
    bands = [(0, 5, (1, 0, 0)), (5, 20, (0, 1, 1)),
             (20, 55, (1, 1, 0)), (55, 105, (0, 1, 0)),
             (105, 170, (1, 1, 1)), (170, 250, (0, 0, 1)),
             (250, 360, (0, 0, 0))]
    h = 1 / 360
    fig, ax = plt.subplots()
    for lo, hi, color in bands:
        ax.fill_between([lo, hi], 0, [h, h], color=color)
    ax.plot([0, 360], [h, h], "k", linewidth=2)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 3e-3)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "UniformExercises_04.png")

    # 5: f black with purple area over [0, 20]
    fig, ax = plt.subplots()
    xs5 = np.linspace(0, 360, 400)
    ax.plot(xs5, np.full_like(xs5, h), "k", linewidth=2)
    ax.fill_between([0, 20], 0, [h, h], color=(0.7, 0, 0.6))
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 3.2e-3)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "UniformExercises_05.png")

    # 6: renormalized conditional density with green areas
    g6 = (1 / 360) / (280 / 360)
    fig, ax = plt.subplots()
    for lo, hi in ((0, 170), (250, 360)):
        ax.plot([lo, hi], [g6, g6], "k", linewidth=1.6)
    ax.plot([170, 250], [0, 0], "k", linewidth=1.6)
    for lo, hi in ((0, 20), (55, 170)):
        ax.fill_between([lo, hi], 0, [g6, g6],
                        color=(0.3, 0.5, 0.2))
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 4.5e-3)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "UniformExercises_06.png")


def mercerkarhunenloeve():
    """stats/MercerKarhunenLoeve — kernel eigenfunctions and KL."""
    # squared-exponential kernel on [-1, 1]
    n = 400
    xs = np.linspace(-1, 1, n)
    dx = xs[1] - xs[0]
    K = np.exp(-((xs[:, None] - xs[None, :]) ** 2) / (2 * 0.3**2))
    lam, V = np.linalg.eigh(K * dx)
    order = np.argsort(-lam)
    lam = lam[order]
    V = V[:, order] / np.sqrt(dx)

    fig, ax = plt.subplots()
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for j, k in enumerate((0, 1, 4, 9)):
        ax.plot(xs, V[:, k] * np.sign(V[np.argmax(np.abs(V[:, k])), k]),
                color=cyc[j], linewidth=2.0)
    ax.set_xlabel("x")
    ax.set_title("First four Mercer eigenfunctions", fontsize=10)
    save(fig, "MercerKarhunenLoeve_01.png")

    fig, ax = plt.subplots()
    ax.semilogy(np.arange(1, 31), np.maximum(lam[:30], 1e-18), ".",
                markersize=9, color=CHEBFUN_BLUE)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("eigenvalues of the kernel", fontsize=10)
    save(fig, "MercerKarhunenLoeve_02.png")

    # KL expansions of random draws with increasing truncation
    rng = np.random.default_rng(0)
    xi = rng.standard_normal(60)
    for j, m in enumerate((5, 20), 3):
        path = (V[:, :m] * np.sqrt(np.maximum(lam[:m], 0))) @ xi[:m]
        fig, ax = plt.subplots()
        ax.plot(xs, path, color=CHEBFUN_BLUE, linewidth=1.4)
        ax.set_title(f"KL sample path, {m} modes", fontsize=10)
        ax.grid(True, alpha=0.4, linewidth=0.4)
        save(fig, f"MercerKarhunenLoeve_{j:02d}.png")

    # variance captured
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, 31), np.cumsum(lam[:30]) / np.sum(lam),
            ".-", markersize=6, linewidth=0.9, color=CHEBFUN_BLUE)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("fraction of variance captured", fontsize=10)
    save(fig, "MercerKarhunenLoeve_05.png")


def smoothrandomwalk():
    """stats/SmoothRandomWalk — cumsum of complex random functions."""
    for j, dx in enumerate((0.1, 0.025, 0.00625, 0.0015625), 1):
        fx = randnfun_np(dx, (0, 1), 1)
        fy = randnfun_np(dx, (0, 1), 2)
        ts = np.linspace(0, 1, 6000)
        # normalized like 'big': amplitude ~ 1/sqrt(dx)
        gx = np.concatenate([[0], np.cumsum(fx(ts)[:-1] * np.diff(ts))]
                            ) / np.sqrt(dx)
        gy = np.concatenate([[0], np.cumsum(fy(ts)[:-1] * np.diff(ts))]
                            ) / np.sqrt(dx)
        fig, ax = plt.subplots()
        ax.plot(gx, gy, color=CHEBFUN_BLUE, linewidth=0.7)
        ax.plot([gx[0]], [gy[0]], ".r", markersize=10)
        ax.set_aspect("equal")
        ax.set_title(f"smooth random walk, dx = {dx:g}", fontsize=10)
        save(fig, f"SmoothRandomWalk_{j:02d}.png")


def bayesiangradebook():
    """stats/BayesianGradebook — belief updates about a grade."""
    theta = np.linspace(0, 1, 1200)

    def phi(mu, sigma):
        return np.exp(-(((theta - mu) / sigma) ** 2) / 2)

    prior = phi(0.7, 0.3)
    prior /= np.trapezoid(prior, theta)

    fig, ax = plt.subplots()
    ax.plot(theta, prior, color=CHEBFUN_BLUE, linewidth=2.0)
    ax.set_xlabel("theta")
    ax.set_title("prior", fontsize=10)
    save(fig, "BayesianGradebook_01.png")

    # observe a score x ~ theta with noise: likelihood, posterior
    x_obs = 0.55
    like = phi(x_obs, 0.15)
    belief = prior * like
    belief /= np.trapezoid(belief, theta)
    fig, ax = plt.subplots()
    ax.plot(theta, belief, linewidth=2.0, color=CHEBFUN_BLUE)
    ax.set_xlabel("theta")
    ax.set_ylabel("P(theta|x)")
    save(fig, "BayesianGradebook_02.png")

    # sequential updates over several scores
    fig, ax = plt.subplots()
    belief_k = prior.copy()
    scores = (0.55, 0.62, 0.58, 0.65)
    for x_o in scores:
        belief_k = belief_k * phi(x_o, 0.15)
        belief_k /= np.trapezoid(belief_k, theta)
        ax.plot(theta, belief_k, linewidth=1.2)
    ax.set_xlabel("theta")
    ax.set_title("posterior after each of four scores", fontsize=10)
    save(fig, "BayesianGradebook_03.png")

    fig, ax = plt.subplots()
    means = []
    belief_k = prior.copy()
    for x_o in scores:
        belief_k = belief_k * phi(x_o, 0.15)
        belief_k /= np.trapezoid(belief_k, theta)
        means.append(np.trapezoid(theta * belief_k, theta))
    ax.plot(range(1, 5), means, ".-", markersize=9, linewidth=1.1,
            color=CHEBFUN_BLUE)
    ax.axhline(np.mean(scores), color="r", linewidth=0.7,
               linestyle="--")
    ax.set_xlabel("number of scores")
    ax.set_title("posterior mean shrinks toward the data",
                 fontsize=10)
    save(fig, "BayesianGradebook_04.png")


def randomsurf():
    """stats/RandomSurf — random smooth surface + paraboloid."""
    from chebfunjax.plotting import PARULA

    rng = np.random.default_rng(3)
    n = 200
    xs = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(xs, xs)
    m = 12
    F = np.zeros_like(X)
    for _ in range(60):
        kx, ky = rng.integers(0, m, 2)
        a, b = rng.standard_normal(2)
        F += (a * np.cos(PI * (kx * X + ky * Y) / 2)
              + b * np.sin(PI * (kx * X + ky * Y) / 2)) \
            * np.exp(-(kx**2 + ky**2) / 18)
    F *= 0.8

    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, F, levels=[F.min(), 0, F.max()],
                     colors=["white", "black"])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("zebra plot of a random function", fontsize=10)
    save(fig, "RandomSurf_01.png")

    G = F + 4 * (X**2 + Y**2)
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, G, levels=[G.min(), 2, G.max()],
                     colors=["white", "black"])
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "RandomSurf_02.png")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, G, cmap=PARULA, rstride=2, cstride=2,
                    linewidth=0)
    ax.set_zlim(-10, 10)
    ax.view_init(elev=60, azim=-90)
    ax.axis("off")
    save(fig, "RandomSurf_03.png")


def randompolynomials():
    """stats/RandomPolynomials — Kac polynomial root clustering."""
    rng = np.random.default_rng(1)
    n = 80
    fig, ax = plt.subplots()
    for _ in range(30):
        c = rng.standard_normal(n + 1)
        r = np.roots(c)
        ax.plot(np.real(r), np.imag(r), ".", color=CHEBFUN_BLUE,
                markersize=2)
    th = np.linspace(0, 2 * PI, 200)
    ax.plot(np.cos(th), np.sin(th), "r-", linewidth=0.8)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_title("Kac polynomial roots cluster at |z| = 1",
                 fontsize=10)
    save(fig, "RandomPolynomials_01.png")

    # random Chebyshev-coefficient polynomials: roots fill [-1, 1]
    fig, ax = plt.subplots()
    from numpy.polynomial import chebyshev as npcheb

    for _ in range(30):
        c = rng.standard_normal(n + 1)
        r = npcheb.chebroots(c)
        r = r[np.abs(np.imag(r)) < 2]
        ax.plot(np.real(r), np.imag(r), ".", color=ORANGE,
                markersize=2)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_title("random Chebyshev series: real roots fill [-1,1]",
                 fontsize=9)
    save(fig, "RandomPolynomials_02.png")

    # density of real roots
    all_real = []
    for _ in range(300):
        c = rng.standard_normal(n + 1)
        r = npcheb.chebroots(c)
        all_real.extend(np.real(r[np.abs(np.imag(r)) < 1e-8]))
    all_real = np.asarray(all_real)
    all_real = all_real[np.abs(all_real) <= 1]
    fig, ax = plt.subplots()
    ax.hist(all_real, bins=40, density=True,
            color=(0.2, 0.15, 0.5), edgecolor=(0.95, 0.95, 0.6),
            linewidth=0.5)
    xs = np.linspace(-0.999, 0.999, 400)
    ax.plot(xs, 1 / (PI * np.sqrt(1 - xs**2)) * 0.9, "r",
            linewidth=1.2)
    ax.set_title("density of real roots", fontsize=10)
    save(fig, "RandomPolynomials_03.png")


def randommaxima():
    """stats/RandomMaxima — local maxima of random functions."""
    for j, dom_end in enumerate((20, 40), 1):
        f = randnfun_np(1.0, (0, dom_end), 0)
        ts = np.linspace(0, dom_end, 4000)
        fv = f(ts)
        # local maxima via sign changes of the derivative
        d = np.diff(fv)
        idx = np.nonzero((d[:-1] > 0) & (d[1:] <= 0))[0] + 1
        fig, ax = plt.subplots()
        ax.plot(ts, fv, "k", linewidth=1.0)
        ax.plot(ts[idx], fv[idx], ".r", markersize=9)
        ax.grid(True, alpha=0.4, linewidth=0.4)
        ax.set_title(f"{len(idx)} local maxima on [0, {dom_end}]",
                     fontsize=10)
        save(fig, f"RandomMaxima_{j:02d}.png")

    # spacing statistics
    f = randnfun_np(1.0, (0, 400), 4)
    ts = np.linspace(0, 400, 40000)
    fv = f(ts)
    d = np.diff(fv)
    idx = np.nonzero((d[:-1] > 0) & (d[1:] <= 0))[0] + 1
    gaps = np.diff(ts[idx])
    fig, ax = plt.subplots()
    ax.hist(gaps, bins=30, density=True, color=(0.2, 0.15, 0.5),
            edgecolor=(0.95, 0.95, 0.6), linewidth=0.5)
    ax.set_title(f"spacing of {len(idx)} maxima (mean "
                 f"{gaps.mean():.2f})", fontsize=10)
    save(fig, "RandomMaxima_03.png")


def histogram():
    """stats/Histogram — histogram of a chebfun's values."""
    dom = (0.0, 10.0)
    f = cj.chebfun(lambda x: x / 3 + jnp.cos(2 * x)
                   + 0.5 * jnp.sin(x**2) + 0.2 * jnp.sin(27 * x),
                   domain=list(dom))
    xs = np.linspace(*dom, 6000)
    fv = np.asarray(f(jnp.asarray(xs)))

    fig, ax = plt.subplots()
    ax.plot(xs, fv, color=CHEBFUN_BLUE, linewidth=0.8)
    save(fig, "Histogram_01.png")

    # per-unit-interval means as red steps over f (MATLAB hist(f))
    fig, ax = plt.subplots()
    ax.plot(xs, fv, color=CHEBFUN_BLUE, linewidth=0.8)
    prev_mean = None
    for k in range(10):
        m = (xs >= k) & (xs < k + 1)
        mk = fv[m].mean()
        ax.plot([k, k + 1], [mk, mk], "r", linewidth=2.0)
        if prev_mean is not None:
            ax.plot([k, k], [prev_mean, mk], ":r", linewidth=0.8)
        prev_mean = mk
    save(fig, "Histogram_02.png")

    # finer edges: the step-mean curve in red alone
    fig, ax = plt.subplots()
    edges = np.linspace(0, 10, 41)
    means = [fv[(xs >= a_) & (xs < b_)].mean()
             for a_, b_ in zip(edges[:-1], edges[1:])]
    ax.step(edges[:-1], means, "r", linewidth=2.0, where="post")
    save(fig, "Histogram_03.png")


def bivariatenormaldistribution():
    """stats/BivariateNormalDistribution — marginals, conditionals."""
    from chebfunjax.plotting import PARULA

    rho = 0.6
    xs = np.linspace(-2, 2, 300)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    det = 1 - rho**2
    P = np.exp(-(X**2 - 2 * rho * X * Y + Y**2) / (2 * det)) / (
        2 * PI * np.sqrt(det))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, P, cmap=PARULA, rstride=2, cstride=2,
                    linewidth=0)
    save(fig, "BivariateNormalDistribution_01.png")

    px = np.trapezoid(P, xs, axis=1)
    fig, ax = plt.subplots()
    ax.plot(xs, px, color=CHEBFUN_BLUE, linewidth=1.6)
    ax.set_title("Marginal distribution", fontsize=10)
    save(fig, "BivariateNormalDistribution_02.png")

    # conditional pdf p(y|x) as a surface
    Pc = P / px[:, None]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Pc, cmap=PARULA, rstride=2, cstride=2,
                    linewidth=0)
    save(fig, "BivariateNormalDistribution_03.png")


def normalexercises():
    """stats/NormalExercises — areas under normal-derived densities."""
    xs = np.linspace(0, 4, 800)
    # density of |X| for X ~ N(0,1): half-normal
    halfn = np.sqrt(2 / PI) * np.exp(-(xs**2) / 2)
    fig, ax = plt.subplots()
    ax.fill_between(xs, 0, halfn, color=(0.3, 0.9, 0.4))
    ax.plot(xs, halfn, "k", linewidth=1.2)
    save(fig, "NormalExercises_01.png")

    # density of X^2: chi-squared with 1 dof
    xs2 = np.linspace(1e-3, 4, 800)
    chi1 = np.exp(-xs2 / 2) / np.sqrt(2 * PI * xs2)
    fig, ax = plt.subplots()
    ax.fill_between(xs2, 0, chi1, color=(0.9, 0.3, 0.4))
    ax.plot(xs2, chi1, "k", linewidth=1.2)
    ax.set_ylim(0, 1.5)
    save(fig, "NormalExercises_02.png")


def leastsquares():
    """stats/LeastSquares — polynomial least squares to noisy data."""
    rng = np.random.default_rng(0)
    npts = 100
    x = np.linspace(-1, 1, npts)
    y = 1.0 / (1 + 25 * x**2) + 1e-1 * rng.standard_normal(npts)
    V = np.polynomial.chebyshev.chebvander(x, 10)
    c, *_ = np.linalg.lstsq(V, y, rcond=None)
    xs = np.linspace(-1, 1, 1500)
    fig, ax = plt.subplots()
    ax.plot(x, y, ".k", markersize=5)
    ax.plot(xs, np.polynomial.chebyshev.chebval(xs, c),
            color=CHEBFUN_BLUE, linewidth=1.6)
    ax.set_title("degree-10 least-squares fit to noisy Runge data",
                 fontsize=9)
    save(fig, "LeastSquares_01.png")

    # continuous polyfit of a jumpy function
    def f_np(t):
        return np.abs(t + 0.2) - 0.5 * np.sign(t - 0.5)

    gl_x, gl_w = np.polynomial.legendre.leggauss(600)
    from numpy.polynomial import legendre as npleg

    cleg = np.array([(2 * k + 1) / 2 * np.sum(
        gl_w * f_np(gl_x) * npleg.legval(gl_x, np.eye(11)[k]))
        for k in range(11)])
    fig, ax = plt.subplots()
    ax.plot(xs, f_np(xs), "k", linewidth=1.2)
    ax.plot(xs, npleg.legval(xs, cleg), color=ORANGE, linewidth=1.4)
    ax.set_title("continuous L2 polyfit of a jumpy chebfun",
                 fontsize=9)
    save(fig, "LeastSquares_02.png")


PAGES = {
    "UniformExercises": uniformexercises,
    "MercerKarhunenLoeve": mercerkarhunenloeve,
    "SmoothRandomWalk": smoothrandomwalk,
    "BayesianGradebook": bayesiangradebook,
    "RandomSurf": randomsurf,
    "RandomPolynomials": randompolynomials,
    "RandomMaxima": randommaxima,
    "Histogram": histogram,
    "BivariateNormalDistribution": bivariatenormaldistribution,
    "NormalExercises": normalexercises,
    "LeastSquares": leastsquares,
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
