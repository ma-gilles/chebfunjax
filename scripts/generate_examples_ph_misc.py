"""Generate genuine example-plot placeholders for a few misc chebfun.org pages.

New standalone generator (does not edit any existing generator). Ports:
  geom/ConstantWidth_01.png   -- zero curve of a chebfun2, filled copper
  geom/Ellipse_01.png         -- parametric ellipse, axis equal
  geom/TwoCircles_01.png      -- overlap-lens of two circular arcs, red fill
  approx3/Tolerance_01.png    -- plotcoeffs of the rows of a loose-tol chebfun3

Faithful ports of the MATLAB sources on chebfun.org.  CompactingColloids
(temp/) is produced by the sibling generator ``generate_examples_ph_pde.py``
(it needs pde15s general function-form Robin/flux boundary conditions, now
supported by ``chebfunjax.chebfun1d.pde15s``).
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

OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
REF_ROOT = ("/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/refs/"
            "docs/images")
PI = float(np.pi)

# MATLAB default color order (used by plotcoeffs on a quasimatrix).
MATLAB_COLORS = [
    (0.0000, 0.4470, 0.7410),
    (0.8500, 0.3250, 0.0980),
    (0.9290, 0.6940, 0.1250),
    (0.4940, 0.1840, 0.5560),
    (0.4660, 0.6740, 0.1880),
    (0.3010, 0.7450, 0.9330),
    (0.6350, 0.0780, 0.1840),
]


def save(fig, cat, name):
    from PIL import Image

    ref_path = os.path.join(REF_ROOT, cat, name)
    size = Image.open(ref_path).size if os.path.exists(ref_path) else (600, 270)
    save_chebfun_figure(fig, os.path.join(OUT_ROOT, cat, name), size=size)
    plt.close(fig)
    print(f"  {cat}/{name} saved")


def constantwidth():
    """geom/ConstantWidth -- a curve of constant width as the zero set of a
    degree-8 bivariate polynomial (Rabinowitz).  MATLAB:

        pc = chebfun2(p,[-11 11 -11 11]);  r = roots(pc);
        copper = [.722 .451 .20];
        fill(real(r),imag(r),copper)
        axis(12*[-1 1 -1 1]), axis square, grid on
    """
    r2 = lambda x, y: x ** 2 + y ** 2
    xy = lambda x, y: x ** 2 - 3 * y ** 2

    def p(x, y):
        R = r2(x, y)
        X = xy(x, y)
        return (R ** 4 - 45 * R ** 3 - 41283 * R ** 2 + 7950960 * R
                + 16 * X ** 3 + 48 * R * X ** 2
                + x * X * (16 * R ** 2 - 5544 * R + 266382) - 373248000)

    pc = cj.chebfun2(p, [-11, 11, -11, 11])
    contours = pc.roots()
    copper = (0.722, 0.451, 0.20)

    fig, ax = plt.subplots()
    for c in contours:
        tt = np.linspace(-1.0, 1.0, 2000)
        z = np.asarray(c(tt))
        ax.fill(np.real(z), np.imag(z), facecolor=copper,
                edgecolor="k", linewidth=1.0)
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([-10, -5, 0, 5, 10])
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "geom", "ConstantWidth_01.png")


def ellipse():
    """geom/Ellipse -- a parametric ellipse.  MATLAB:

        theta = chebfun(@(theta) theta,[0,2*pi]);
        x = (0.5/pi)*cos(theta);  y = (0.4/pi)*sin(theta);
        plot(x,y,'-','LineWidth',2), axis equal
    """
    theta = cj.chebfun(lambda t: t, domain=[0.0, 2 * PI])
    x = (0.5 / PI) * cj.cos(theta)
    y = (0.4 / PI) * cj.sin(theta)
    tt = jnp.linspace(0.0, 2 * PI, 2000)
    xx = np.asarray(x(tt))
    yy = np.asarray(y(tt))

    fig, ax = plt.subplots()
    ax.plot(xx, yy, "-", color=CHEBFUN_BLUE, linewidth=2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.27, 0.27)
    ax.set_ylim(-0.14, 0.14)
    ax.set_xticks(np.arange(-0.25, 0.26, 0.05))
    ax.set_yticks(np.arange(-0.1, 0.11, 0.05))
    save(fig, "geom", "Ellipse_01.png")


def twocircles():
    """geom/TwoCircles -- overlap lens of two circular arcs.  MATLAB:

        bigcircle    = chebfun(@(x) sqrt(4-(x-1).^2),'splitting','on');
        littlecircle = chebfun(@(x) 2-sqrt(1-(x+1).^2),[-1,0],'splitting','on');
        plot([-1 1 1 -1 -1],[0 0 2 2 0],'k'), hold on, axis equal
        x = roots( bigcircle{-1,0} - littlecircle );
        fill(join(t,t_reverse),join(littlecircle(t),bigcircle(t_reverse)),'r')
        plot(bigcircle,'k',littlecircle,'k'), axis([-1 1 0 2]), hold off
        set(gca,'xtick',-1:1,'ytick',0:2)
    """
    big = lambda x: np.sqrt(4.0 - (x - 1.0) ** 2)      # x in [-1, 1]
    little = lambda x: 2.0 - np.sqrt(1.0 - (x + 1.0) ** 2)  # x in [-1, 0]

    # roots of (big - little) on [-1,0].  The two circle arcs each carry a
    # sqrt endpoint singularity, so we bracket-and-refine numerically rather
    # than build singular chebfuns (MATLAB uses 'splitting','on').
    from scipy.optimize import brentq
    d = lambda x: big(x) - little(x)
    xs = np.linspace(-1.0 + 1e-9, -1e-9, 4000)
    dv = d(xs)
    sgn = np.where(np.diff(np.sign(dv)))[0]
    rts = sorted(brentq(d, xs[i], xs[i + 1]) for i in sgn)
    x1, x2 = float(rts[0]), float(rts[-1])

    fig, ax = plt.subplots()
    # black bounding square
    ax.plot([-1, 1, 1, -1, -1], [0, 0, 2, 2, 0], "k", linewidth=1.0)
    # red overlap lens: little (bottom) from x1->x2, big (top) back x2->x1
    xf = np.linspace(x1, x2, 400)
    poly_x = np.concatenate([xf, xf[::-1]])
    poly_y = np.concatenate([little(xf), big(xf[::-1])])
    ax.fill(poly_x, poly_y, "r", edgecolor="none")
    # the two arcs in black
    xb = np.linspace(-1.0, 1.0, 600)
    ax.plot(xb, big(xb), "k", linewidth=1.0)
    xl = np.linspace(-1.0, 0.0, 400)
    ax.plot(xl, little(xl), "k", linewidth=1.0)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([0, 1, 2])
    save(fig, "geom", "TwoCircles_01.png")


def tolerance():
    """approx3/Tolerance -- Chebyshev coefficients of the rows of a chebfun3
    built with a loosened tolerance.  MATLAB (chebfun3eps 1e-4):

        g = exp(sin(10*x.*y.*z + exp(x.*y.*z)));
        plotcoeffs(g.rows), ylim([3e-6 10]), title('Chebyshev coefficients')
    """
    from chebfunjax.chebfun3d import chebfun3

    g = chebfun3(
        lambda x, y, z: jnp.exp(jnp.sin(10 * x * y * z
                                        + jnp.exp(x * y * z))),
        tol=1e-3)

    fig, ax = plt.subplots()
    for k, row in enumerate(g.rows):
        c = np.asarray(row.coeffs)
        # MATLAB's chebfun3 stores the row factors as unit-norm columns (the
        # scale lives in the core tensor); cj keeps them un-normalised, so
        # rescale each row to a unit coefficient 2-norm before plotcoeffs.
        c = np.abs(c) / np.linalg.norm(c)
        col = MATLAB_COLORS[k % len(MATLAB_COLORS)]
        ax.semilogy(np.arange(c.shape[0]), c, color=col, linewidth=1.2)
    ax.set_ylim(3e-6, 10)
    ax.set_yticks([1e-4, 1e-2, 1e0])
    ax.set_xlim(-0.9, 35.9)
    ax.set_xticks(np.arange(0, 36, 5))
    ax.set_title("Chebyshev coefficients")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "approx3", "Tolerance_01.png")


PAGES = {
    "ConstantWidth": constantwidth,
    "Ellipse": ellipse,
    "TwoCircles": twocircles,
    "Tolerance": tolerance,
    # CompactingColloids -> see generate_examples_ph_pde.py (pde15s Robin BC).
}


if __name__ == "__main__":
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, fn in PAGES.items():
        if flt.lower() in name.lower():
            print(f"[{name}]")
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
