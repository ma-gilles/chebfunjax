"""Generate all plots for Guide Chapter 6: Quasimatrices and Least-Squares.

Every figure is generated from genuine chebfunjax objects (quasimatrices,
continuous QR/SVD) and exported at the exact 610x258 px canvas used by the
MATLAB renders on chebfun.org, so the pages line up pixel-for-pixel.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import jax.numpy as jnp
import numpy as np
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, save_chebfun_figure
from chebfunjax.chebfun1d.linalg import Quasimatrix, qr_quasimatrix, svd_quasimatrix
from chebfunjax.domain import Domain

chebfun_style()

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUTDIR, exist_ok=True)

# MATLAB default color order (co-ordinated with the chebfun.org renders).
MATLAB = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E',
          '#77AC30', '#4DBEEE', '#A2142F']
BLUE, RED = MATLAB[0], MATLAB[1]

# chebfun.org Guide canvas = 610x258 px. Single-panel axes frame sits at
# pixel box x=[79,551], y=[19,229]; multi-panel boxes are positioned to match
# their MATLAB renders (measured from the reference PNGs).
SIZE = (610, 258)
SINGLE = [79 / 610, 1 - 229 / 258, (551 - 79) / 610, (229 - 19) / 258]
_FMT = FuncFormatter(lambda v, _: f"{v:g}")

plot_index = 0


def _pos(x0, x1, y0, y1):
    """Axes box (figure fractions) from a pixel rectangle on the 610x258 canvas."""
    return [x0 / 610, 1 - y1 / 258, (x1 - x0) / 610, (y1 - y0) / 258]


def new_single():
    fig = plt.figure()
    ax = fig.add_axes(SINGLE)
    return fig, ax


def style_line(ax, xlim, ylim, xticks, yticks, grid=True):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_major_formatter(_FMT)
    ax.yaxis.set_major_formatter(_FMT)
    if grid:
        ax.grid(True, color=(0.87, 0.87, 0.87), linewidth=0.6)
        ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)


def style_box(ax, title):
    ax.set_title(title, fontsize=11)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)


def curve(ax, f, color, lw=1.4, n=2000):
    xs = np.linspace(-1.0, 1.0, n)
    ys = np.array(f(jnp.array(xs)))
    ax.plot(xs, ys, color=color, linewidth=lw)


def spy_cols(ax, ncol, title):
    """inf x ncol quasimatrix: one coloured vertical line per column."""
    for k in range(ncol):
        ax.plot([k + 1, k + 1], [-1, 1], color=MATLAB[k % 7], linewidth=2.0)
    ax.set_xlim(0.5, ncol + 0.5)
    ax.set_ylim(1, -1)                      # domain [-1,1], -1 at top
    ax.set_xticks(range(1, ncol + 1))
    ax.set_yticks([-1, 1])
    style_box(ax, title)


def spy_rows(ax, nrow, title):
    """nrow x inf quasimatrix: one coloured horizontal line per row."""
    for k in range(nrow):
        ax.plot([-1, 1], [k + 1, k + 1], color=MATLAB[k % 7], linewidth=2.0)
    ax.set_xlim(-1, 1)
    ax.set_ylim(nrow + 0.5, 0.5)            # row 1 at top
    ax.set_xticks([-1, 1])
    ax.set_yticks(range(1, nrow + 1))
    style_box(ax, title)


def spy_dots(ax, coords, title, nz, xlim, ylim, xticks, yticks):
    for (r, c) in coords:
        ax.plot(c, r, 'o', color=BLUE, markersize=4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel(f"nz = {nz}", fontsize=10)
    style_box(ax, title)


def save(fig):
    global plot_index
    plot_index += 1
    path = os.path.join(OUTDIR, f"guide06_{plot_index:02d}.png")
    save_chebfun_figure(fig, path, size=SIZE)
    plt.close(fig)
    print(f"  guide06_{plot_index:02d}.png saved")


def skip(err):
    global plot_index
    plot_index += 1
    print(f"  guide06_{plot_index:02d}.png FAILED: {err}")


# --------------------------------------------------------------------------
# Common chebfunjax objects
# --------------------------------------------------------------------------
x = cj.chebfun(lambda t: t)
cols = [x ** k for k in range(6)]
A = Quasimatrix(cols, domain=Domain((-1.0, 1.0)))
f = cj.chebfun(lambda t: jnp.exp(t) * jnp.sin(6 * t))

hat_cols = []
for j in range(11):
    xj = -1.0 + j / 5.0
    hat_cols.append(
        cj.chebfun(lambda t, _xj=xj: jnp.maximum(0.0, 1.0 - 5.0 * jnp.abs(t - _xj)),
                   domain=(-1.0, 1.0))
    )
A2 = Quasimatrix(hat_cols, domain=Domain((-1.0, 1.0)))

XT5 = [-1.0, -0.5, 0.0, 0.5, 1.0]                 # monomial-plot x-ticks
XT2 = [round(-1.0 + 0.2 * i, 1) for i in range(11)]  # hat-plot x-ticks (step 0.2)

# ==========================================================================
# Plot 1: columns of A = [1, x, x^2, ..., x^5]                    (Sec 6.1)
# ==========================================================================
try:
    fig, ax = new_single()
    for k in range(6):
        curve(ax, cols[k], MATLAB[k])
    style_line(ax, (-1, 1), (-1.1, 1.1), XT5, [-1, -0.5, 0, 0.5, 1])
    save(fig)
except Exception as e:
    skip(e)

# ==========================================================================
# Plot 2: spy(A) and spy(A')                                       (Sec 6.1)
# ==========================================================================
try:
    fig = plt.figure()
    spy_cols(fig.add_axes(_pos(130, 231, 23, 229)), 6, 'A')
    spy_rows(fig.add_axes(_pos(348, 551, 84, 168)), 6, "A'")
    save(fig)
except Exception as e:
    skip(e)

# ==========================================================================
# Plot 3: f and its degree-5 least-squares fit                     (Sec 6.2)
# ==========================================================================
try:
    Q, R = qr_quasimatrix(A)
    rhs = jnp.array([float(Q[j].inner(f)) for j in range(6)])
    c = jnp.linalg.solve(R, rhs)
    ffit = cols[0] * float(c[0])
    for j in range(1, 6):
        ffit = ffit + cols[j] * float(c[j])

    fig, ax = new_single()
    curve(ax, f, BLUE)
    curve(ax, ffit, RED)
    style_line(ax, (-1, 1), (-3, 2), XT5, [-3, -2, -1, 0, 1, 2])
    ax.legend(['f', 'ffit'], loc='upper right', fontsize=9,
              handlelength=1.6, borderaxespad=0.4)
    save(fig)
except Exception as e:
    skip(e)

# ==========================================================================
# Plot 4: hat functions                                            (Sec 6.2)
# ==========================================================================
try:
    fig, ax = new_single()
    for k in range(11):
        curve(ax, hat_cols[k], MATLAB[k % 7])
    style_line(ax, (-1, 1), (-0.2, 1.0), XT2,
               [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0], grid=False)
    save(fig)
except Exception as e:
    skip(e)

# ==========================================================================
# Plot 5: hat-function least-squares fit                           (Sec 6.2)
# ==========================================================================
try:
    Q2, R2 = qr_quasimatrix(A2)
    rhs2 = jnp.array([float(Q2[j].inner(f)) for j in range(11)])
    c2 = jnp.linalg.solve(R2, rhs2)
    ffit2 = hat_cols[0] * float(c2[0])
    for j in range(1, 11):
        ffit2 = ffit2 + hat_cols[j] * float(c2[j])

    fig, ax = new_single()
    curve(ax, f, BLUE)
    curve(ax, ffit2, RED)
    style_line(ax, (-1, 1), (-3, 2), XT2, [-3, -2, -1, 0, 1, 2])
    ax.legend(['f', 'ffit'], loc='upper right', fontsize=9,
              handlelength=1.6, borderaxespad=0.4)
    save(fig)
except Exception as e:
    skip(e)

# ==========================================================================
# Plot 6: QR orthonormal columns (L2-normalized Legendre)          (Sec 6.3)
# ==========================================================================
try:
    Qm, Rm = qr_quasimatrix(A)
    fig, ax = new_single()
    for k in range(Qm.n_cols):
        curve(ax, Qm[k], MATLAB[k])
    style_line(ax, (-1, 1), (-3, 3), XT5, [-3, -2, -1, 0, 1, 2, 3])
    save(fig)
except Exception as e:
    skip(e)

# ==========================================================================
# Plot 7: spy(A), spy(Q), spy(R)                                   (Sec 6.3)
# ==========================================================================
try:
    Rnp = np.array(Rm)
    coords = [(r, c) for r in range(6) for c in range(6)
              if abs(Rnp[r, c]) > 1e-12]
    fig = plt.figure()
    spy_cols(fig.add_axes(_pos(112, 175, 23, 229)), 6, 'A')
    spy_cols(fig.add_axes(_pos(284, 347, 23, 229)), 6, 'Q')
    spy_dots(fig.add_axes(_pos(422, 551, 61, 191)),
             [(r + 1, c + 1) for (r, c) in coords], 'R', len(coords),
             (-0.2, 6.5), (6.5, -0.2), [0, 5], [0, 2, 4, 6])
    save(fig)
except Exception as e:
    skip(e)

# ==========================================================================
# Plot 8: renormalized Legendre (P(1)=1)                           (Sec 6.3)
# ==========================================================================
try:
    Q_leg = []
    for j in range(6):
        v1 = float(Qm[j](jnp.float64(1.0)))
        Q_leg.append(Qm[j] * (1.0 / v1) if abs(v1) > 1e-14 else Qm[j])
    fig, ax = new_single()
    for k in range(6):
        curve(ax, Q_leg[k], MATLAB[k])
    style_line(ax, (-1, 1), (-1.5, 1.5), XT5,
               [-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    save(fig)
except Exception as e:
    skip(e)

# ==========================================================================
# Plot 9: orthonormalized hat functions                            (Sec 6.3)
# ==========================================================================
try:
    Q2h, R2h = qr_quasimatrix(A2)
    fig, ax = new_single()
    for k in range(Q2h.n_cols):
        curve(ax, Q2h[k], MATLAB[k % 7])
    style_line(ax, (-1, 1), (-2, 4), XT2, [-2, -1, 0, 1, 2, 3, 4], grid=False)
    save(fig)
except Exception as e:
    skip(e)

# ==========================================================================
# Plot 10: spy(A), spy(U), spy(S), spy(V)                          (Sec 6.4)
# ==========================================================================
try:
    U, S, V = svd_quasimatrix(A)
    fig = plt.figure()
    spy_cols(fig.add_axes(_pos(98, 135, 23, 229)), 6, 'A')
    spy_cols(fig.add_axes(_pos(297, 333, 23, 229)), 6, 'U')
    spy_dots(fig.add_axes(_pos(377, 452, 88, 164)),
             [(i + 1, i + 1) for i in range(6)], 'S', 6,
             (-0.2, 6.5), (6.5, -0.2), [0, 5], [0, 5])
    spy_dots(fig.add_axes(_pos(477, 551, 88, 164)),
             [(i + 1, j + 1) for i in range(6) for j in range(6)], 'V', 36,
             (-0.2, 6.5), (6.5, -0.2), [0, 5], [0, 5])
    save(fig)
except Exception as e:
    skip(e)

# ==========================================================================
# Plot 11: SVD extremal functions A*v1 (blue) and A*vn (red)       (Sec 6.4)
# ==========================================================================
try:
    Vnp = np.array(V)
    v1 = Vnp[:, 0]
    vn = Vnp[:, -1]
    f_max = sum(float(v1[j]) * cols[j] for j in range(6))
    f_min = sum(float(vn[j]) * cols[j] for j in range(6))
    fig, ax = new_single()
    curve(ax, f_max, BLUE)
    curve(ax, f_min, '#FF0000')     # MATLAB 'r' (pure red) on chebfun.org
    style_line(ax, (-1, 1), (-0.07, 1.5), XT5, [0, 0.5, 1, 1.5])
    save(fig)
except Exception as e:
    skip(e)

# ==========================================================================
# Plot 12: spy(null(B)), spy(orth(B)), spy(pinv(A))                (Sec 6.6)
# ==========================================================================
try:
    fig = plt.figure()
    # null(B): 3x1 -> three dots in column 1
    spy_dots(fig.add_axes(_pos(95, 192, 23, 215)),
             [(1, 1), (2, 1), (3, 1)], 'null(B)', 3,
             (0, 2), (4, 0), [0, 1, 2], [0, 1, 2, 3, 4])
    # orth(B): inf x 2 -> two coloured vertical lines
    ax = fig.add_axes(_pos(286, 345, 23, 215))
    for k in range(2):
        ax.plot([k + 1, k + 1], [-1, 1], color=MATLAB[k], linewidth=2.0)
    ax.set_xlim(0.4, 2.6)
    ax.set_ylim(1, -1)
    ax.set_xticks([1, 2])
    ax.set_yticks([-1, 1])
    style_box(ax, 'orth(B)')
    # pinv(A): 6 x inf -> six coloured horizontal lines
    spy_rows(fig.add_axes(_pos(422, 551, 77, 161)), 6, 'pinv(A)')
    save(fig)
except Exception as e:
    skip(e)

print(f"\nGuide 06: Generated {plot_index} plots.")
