"""Generate all plots for Guide Chapter 4: Chebfun and Approximation Theory.

Faithfully translates every figure from Chebfun Guide Chapter 4
(https://www.chebfun.org/docs/guide/guide04.html) to chebfunjax/Python.

Each figure is exported at the exact pixel size of its chebfun.org reference
render (610x258 for the guide figures) with the MATLAB axes-box position
measured from the reference images, so it can be compared pixel-for-pixel
against the reference.  The MATLAB commands that produce each figure are
quoted above each block.

Axis limits and tick locations are set explicitly to the values MATLAB
produced in the reference renders, so the figures do not depend on the
library's automatic tick heuristic.

A few MATLAB Chebfun features used in this chapter are not yet available in
chebfunjax; where that is the case the figure is reproduced with the public
chebfunjax API plus NumPy/matplotlib, and the workaround is noted in a
comment.  In particular the Caratheodory-Fejer command ``cf`` has no
chebfunjax equivalent, so a faithful NumPy port of Chebfun's
``@chebfun/cf.m`` (rational CF) is included below; it consumes the Chebyshev
coefficients produced by chebfunjax (``f.coeffs``).  See the accompanying
report for the full list of library gaps.
"""

import matplotlib

matplotlib.use('Agg')

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
from scipy.linalg import hankel, toeplitz

import chebfunjax as cj
from chebfunjax.domain import Domain
from chebfunjax.plotting import (
    CHEBFUN_BLUE,
    chebfun_style,
    save_chebfun_figure,
)
from chebfunjax.utils.lebesgue import lebesgue_function
from chebfunjax.utils.minimax import minimax
from chebfunjax.utils.polynomials import chebpoly
from chebfunjax.utils.quadrature import chebpts
from chebfunjax.utils.ratapprox import chebpade, ratinterp

chebfun_style()

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT_DIR, exist_ok=True)

REF_SIZE = (610, 258)
BLUE = CHEBFUN_BLUE
RED = 'r'
MAG = 'm'
CYAN = 'c'
BLACK = 'k'

UNIT_X = [-1, -0.5, 0, 0.5, 1]


# ==========================================================================
# Axes-box geometry, measured from the reference renders (610x258).
# ==========================================================================

def _box(L, R, T, B):
    """subplots_adjust dict for a single axes with spine box (L,R,T,B) px."""
    return dict(left=L / 610, right=R / 610, bottom=1 - B / 258, top=1 - T / 258)


def _pos(L, R, T, B):
    """axes position [x0, y0, w, h] for spine box (L,R,T,B) px on 610x258."""
    return [L / 610, 1 - B / 258, (R - L) / 610, (B - T) / 258]


BOX_SINGLE = _box(79, 551, 19, 229)          # single axes, no title
# 1x2 subplots, no subplot title
S2A, S2B = _pos(79, 282, 19, 229), _pos(348, 551, 19, 229)
# 1x2 subplots with subplot title (figs 07, 08)
S2AT, S2BT = _pos(79, 281, 23, 229), _pos(348, 550, 23, 229)
# 2x1 subplots (fig 09)
V1, V2 = _pos(79, 551, 30, 106), _pos(79, 551, 152, 228)
# 2x2 subplots (fig 01)
Q_TL, Q_TR = _pos(79, 282, 19, 106), _pos(348, 551, 19, 106)
Q_BL, Q_BR = _pos(79, 282, 142, 228), _pos(348, 551, 142, 228)


def _gfmt(ax):
    """MATLAB-style compact x tick labels (-1 not -1.0) on linear axes.

    Applied to the x-axis only: several figures have tiny y-values shown with
    a shared 1e-n offset that matches the MATLAB reference, and %g would break
    that, so the y-axis formatter is left untouched.
    """
    from matplotlib.ticker import FuncFormatter
    if ax.get_xscale() == 'linear':
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def _save(fig, idx):
    fig.set_facecolor('white')
    path = os.path.join(OUT_DIR, f'guide04_{idx:02d}.png')
    save_chebfun_figure(fig, path, size=REF_SIZE)
    plt.close(fig)
    print(f"  guide04_{idx:02d}.png saved")


def _finish_single(fig, ax, idx, *, xlim=None, ylim=None, xticks=None,
                   yticks=None, title=None, grid=False, which='major'):
    fig.subplots_adjust(**BOX_SINGLE)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if title is not None:
        ax.set_title(title, fontsize=10)
    if grid:
        ax.grid(True, which=which, color='0.85', linewidth=0.6)
        ax.set_axisbelow(True)
    _gfmt(ax)
    _save(fig, idx)


def _style_sub(ax, *, xlim=None, ylim=None, xticks=None, yticks=None,
               title=None, grid=False, which='major'):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if title is not None:
        ax.set_title(title, fontsize=10)
    if grid:
        ax.grid(True, which=which, color='0.85', linewidth=0.6)
        ax.set_axisbelow(True)
    _gfmt(ax)


# ==========================================================================
# NumPy port of Chebfun @chebfun/cf.m (rational Caratheodory-Fejer).
# chebfunjax has no `cf`; this reproduces it from the Chebyshev coefficients
# returned by chebfunjax (f.coeffs).  Validated against the known CF errors:
# exp(5,5) ~ 1e-13 (11 equioscillations), |x-.3|(5,5) ~ 6e-3, tanh(40,4) ~ 1e-10.
# ==========================================================================

_EPS = np.finfo(float).eps


def _clenshaw(coeffs, x):
    x = np.asarray(x, dtype=float)
    c = np.asarray(coeffs, dtype=float)
    bk1 = np.zeros_like(x); bk2 = np.zeros_like(x)
    for k in range(len(c) - 1, 0, -1):
        bk = c[k] + 2 * x * bk1 - bk2
        bk2 = bk1; bk1 = bk
    return c[0] + x * bk1 - bk2


def _chebcoeffs_func(func, K):
    """Chebyshev coeffs c_0..c_K of func on [-1, 1] (K+1 coeffs, ascending)."""
    n = K + 1
    if n == 1:
        return np.array([float(func(np.array([0.0]))[0])])
    x = np.cos(np.pi * np.arange(n) / (n - 1))[::-1]
    v = np.asarray(func(x), dtype=float)
    tmp = np.concatenate([v[::-1], v[1:-1]])
    c = np.fft.ifft(tmp).real[:n].copy()
    c[1:-1] *= 2.0
    return c


def _cf_getBlock(a, m, n, M):
    tol = 1e-14
    if n > M + m + 1:
        c = np.zeros(n - m - M - 1); nn = M + m + 1
    else:
        c = np.array([]); nn = n
    idx = np.abs(np.arange(m - nn + 1, M + 1))
    c = np.concatenate([c, a[M - idx]])
    w, V = np.linalg.eig(hankel(c))
    order = np.argsort(-np.abs(w))
    S = np.abs(w[order])
    s = w[order[n]]
    u = np.real(V[:, order[n]])
    tmp = np.abs(S - np.abs(s)) < tol
    k = ll = 0
    while (k < n) and tmp[n - k - 1]:
        k += 1
    while ((n + ll + 2) < len(tmp)) and tmp[n + ll + 1]:
        ll += 1
    return s, u, k, ll, (n + ll + 2) == len(tmp)


def cf_rational(a_asc, m, n, M, vscale):
    """Rational CF: return (r_ref, q_ref, s) acting on t in [-1, 1]."""
    a = a_asc[:M + 1][::-1].astype(float).copy()   # descending: a[0]=c_M..a[M]=c_0
    a[M] = 2 * a[M]
    tolfft, maxnfft = 1e-14, 2 ** 17

    even_c, odd_c = a[M - 1::-2], a[M::-2]
    if len(even_c) and np.max(np.abs(even_c)) / vscale < _EPS:
        if not (m % 2 or n % 2):
            m += 1
        elif (m % 2) and (n % 2):
            n -= 1
    elif len(odd_c) and np.max(np.abs(odd_c)) / vscale < _EPS:
        if (m % 2) and not (n % 2):
            m += 1
        elif not (m % 2) and (n % 2):
            n -= 1

    s, u, k, ll, _ = _cf_getBlock(a, m, n, M)
    if (k > 0) or (ll > 0):
        s2, u2, knew, lnew, _ = _cf_getBlock(a, m + ll, n - k, M)
        if (knew > 0) or (lnew > 0):
            n = n + ll
            s, u, k, ll, _ = _cf_getBlock(a, m - k, n, M)
        else:
            n = n - k; s, u = s2, u2

    def _tail(arr, kk):
        L = len(arr)
        return arr[L - kk - 1:L - 1]

    # denominator q from Laurent coefficients
    N = max(2 ** int(np.ceil(np.log2(len(u)))), 256)
    ud = np.polyder(u[::-1])[::-1]
    ac = fft(np.conj(fft(ud, N) / fft(u, N))) / N
    act = np.zeros(N, dtype=complex)
    while (np.linalg.norm(1 - _tail(act, n) / _tail(ac, n), np.inf) > tolfft) and (N < maxnfft):
        act = ac; N *= 2
        ac = fft(np.conj(fft(ud, N) / fft(u, N))) / N
    ac = np.real(ac)
    b = np.ones(n + 1)
    for j in range(1, n + 1):
        b[j] = -np.dot(b[:j], ac[N - j - 1:N - 1]) / j
    z = np.roots(b)
    z = z[np.abs(z) < 1]
    rho = 1.0 / np.max(np.abs(z))
    z = 0.5 * (z + 1.0 / z)

    def q_ref(t):
        t = np.asarray(t, dtype=float)
        num = np.ones_like(t, dtype=complex)
        for zi in z:
            num = num * (t - zi)
        return np.real(num / np.prod(-z))

    # Chebyshev coeffs of Rt from Blaschke product FFT
    v = u[::-1]
    N = max(2 ** int(np.ceil(np.log2(len(u)))), 256)

    def _acR(N):
        return fft(np.exp(2j * np.pi * M * np.arange(N) / N) *
                   np.conj(fft(u, N) / fft(v, N))) / N
    ac = _acR(N)
    act = np.zeros(N, dtype=complex)
    while (
        (np.linalg.norm(1 - act[:m + 1] / ac[:m + 1], np.inf) > tolfft)
        and (np.linalg.norm(1 - act[len(act) - m:] / ac[len(ac) - m:], np.inf) > tolfft)
        and (N < maxnfft)
    ):
        act = ac; N *= 2
        ac = _acR(N)
    ac = s * np.real(ac)
    a_slice = a[M::-1][:m + 1]
    bracket = np.concatenate([[ac[0]], ac[N - 1:N - m - 1:-1]])
    ct = a_slice - ac[:m + 1] - bracket
    s = abs(s)

    # numerator polynomial
    deg = int(np.ceil(np.log(4 / _EPS / (rho - 1)) / np.log(rho)))
    gam = _chebcoeffs_func(lambda t: 1.0 / q_ref(t), deg)[::-1]
    if len(gam) < 2 * m + 1:
        gam = np.concatenate([np.zeros(2 * m + 1 - len(gam)), gam])
    gam = gam[::-1][:2 * m + 1].copy()
    gam[0] = 2 * gam[0]
    G_toe = toeplitz(gam)
    A = G_toe[:m, :m]
    B = G_toe[:m, m:m + 1]
    C = G_toe[:m, 2 * m:m:-1]
    G = A + C - 2 * (B @ B.T) / G_toe[0, 0]
    rhs = -2 * (B * ct[0] / G_toe[0, 0] - ct[m:0:-1].reshape(-1, 1))
    bc = np.linalg.solve(G, rhs)
    bc0 = (ct[0] - (B.T @ bc)[0, 0]) / G_toe[0, 0]
    bc_full = np.concatenate([[bc0], bc[::-1, 0]])

    def r_ref(t):
        return _clenshaw(bc_full, t) / q_ref(t)
    return r_ref, q_ref, s


# ==========================================================================
# Section 4.1  Chebyshev series and interpolants
# ==========================================================================

# --------------------------------------------------------------------------
# Fig 1:  subplot(2,2,k), plot(chebpoly(N)), ylim([-1.5 1.5])  for N=2,3,15,50
# --------------------------------------------------------------------------
try:
    print("Fig 1: T_2, T_3, T_15, T_50")
    fig, axes = plt.subplots(2, 2)
    positions = [Q_TL, Q_TR, Q_BL, Q_BR]
    for ax, N, posn in zip(axes.flat, [2, 3, 15, 50], positions):
        T = cj.Chebfun.from_coeffs(jnp.array(chebpoly(N)))
        xs = np.linspace(-1, 1, 3000)
        ax.plot(xs, np.asarray(T(jnp.array(xs))), '-', color=BLUE, linewidth=1.0)
        ax.set_position(posn)
        _style_sub(ax, xlim=(-1, 1), ylim=(-1.5, 1.5),
                   xticks=UNIT_X, yticks=[-1, 0, 1])
    _save(fig, 1)
except Exception as e:
    print(f"  guide04_01 FAILED: {e}")


# ==========================================================================
# Section 4.3  chebfun(...,N) and the Gibbs phenomenon
# ==========================================================================

def _plot_marked(ax, f, a, b, n, ms=5, dense=4000):
    """MATLAB plot(f,'.-'): dense interpolant line + dots at Chebyshev nodes."""
    xs = np.linspace(a, b, dense)
    ax.plot(xs, np.asarray(f(jnp.array(xs))), '-', color=BLUE, linewidth=1.0)
    xn = np.asarray(chebpts(n))
    if b != 1.0 or a != -1.0:
        xn = 0.5 * (a + b) + 0.5 * (b - a) * xn
    ax.plot(xn, np.asarray(f(jnp.array(xn))), '.', color=BLUE, markersize=ms)


# Fig 2:  f = chebfun('sign(x)',10/20); plot(f,'.-',MS,8), grid on
try:
    print("Fig 2: sign(x), N=10 and 20")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for ax, N, posn in [(ax1, 10, S2A), (ax2, 20, S2B)]:
        f = cj.chebfun(lambda x: jnp.sign(x), n=N)
        _plot_marked(ax, f, -1, 1, N)
        ax.set_position(posn)
        _style_sub(ax, xlim=(-1, 1), ylim=(-1.5, 1.5), xticks=UNIT_X,
                   yticks=[-1.5, -1, -0.5, 0, 0.5, 1, 1.5], grid=True)
    _save(fig, 2)
except Exception as e:
    print(f"  guide04_02 FAILED: {e}")

# Fig 3:  same as fig 2, axis([0 .8 .5 1.5]) and axis([0 .4 .5 1.5])
try:
    print("Fig 3: sign(x) zoomed, N=10 and 20")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for ax, N, xr, xt, posn in [
        (ax1, 10, 0.8, [0, 0.2, 0.4, 0.6, 0.8], S2A),
        (ax2, 20, 0.4, [0, 0.1, 0.2, 0.3, 0.4], S2B)]:
        f = cj.chebfun(lambda x: jnp.sign(x), n=N)
        _plot_marked(ax, f, -1, 1, N)
        ax.set_position(posn)
        _style_sub(ax, xlim=(0, xr), ylim=(0.5, 1.5), xticks=xt,
                   yticks=[0.5, 1, 1.5], grid=True)
    _save(fig, 3)
except Exception as e:
    print(f"  guide04_03 FAILED: {e}")

# Fig 4:  N=100 axis([0 .08 .5 1.5]); N=1000 axis([0 .008 .5 1.5])
try:
    print("Fig 4: sign(x) zoomed, N=100 and 1000")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for ax, N, xr, xt, posn in [
        (ax1, 100, 0.08, [0, 0.02, 0.04, 0.06, 0.08], S2A),
        (ax2, 1000, 0.008, [0, 0.002, 0.004, 0.006, 0.008], S2B)]:
        f = cj.chebfun(lambda x: jnp.sign(x), n=N)
        _plot_marked(ax, f, -1, 1, N, ms=6, dense=8000)
        ax.set_position(posn)
        _style_sub(ax, xlim=(0, xr), ylim=(0.5, 1.5), xticks=xt,
                   yticks=[0.5, 1, 1.5], grid=True)
    _save(fig, 4)
except Exception as e:
    print(f"  guide04_04 FAILED: {e}")


# ==========================================================================
# Section 4.4  Smoothness and rate of convergence
# ==========================================================================

# Fig 5:  f = chebfun('abs(x)',10/20); plot(f,'.-',MS,8), ylim([0 1]), grid on
try:
    print("Fig 5: abs(x), N=10 and 20")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for ax, N, posn in [(ax1, 10, S2A), (ax2, 20, S2B)]:
        f = cj.chebfun(lambda x: jnp.abs(x), n=N)
        _plot_marked(ax, f, -1, 1, N)
        ax.set_position(posn)
        _style_sub(ax, xlim=(-1, 1), ylim=(0, 1), xticks=UNIT_X,
                   yticks=[0, 0.2, 0.4, 0.6, 0.8, 1], grid=True)
    _save(fig, 5)
except Exception as e:
    print(f"  guide04_05 FAILED: {e}")

# Fig 6:  f = chebfun('abs(x)',100/1000); plot(f), ylim([0 1]), grid on
try:
    print("Fig 6: abs(x), N=100 and 1000")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for ax, N, posn in [(ax1, 100, S2A), (ax2, 1000, S2B)]:
        f = cj.chebfun(lambda x: jnp.abs(x), n=N)
        xs = np.linspace(-1, 1, 6000)
        ax.plot(xs, np.asarray(f(jnp.array(xs))), '-', color=BLUE, linewidth=1.0)
        ax.set_position(posn)
        _style_sub(ax, xlim=(-1, 1), ylim=(0, 1), xticks=UNIT_X,
                   yticks=[0, 0.2, 0.4, 0.6, 0.8, 1], grid=True)
    _save(fig, 6)
except Exception as e:
    print(f"  guide04_06 FAILED: {e}")

# Fig 7:  |x|^5 convergence.  loglog(e) + loglog(NN^-5,'--r'); semilogy(e)+...
try:
    print("Fig 7: |x|^5 convergence")
    exact = cj.chebfun(lambda x: jnp.abs(x) ** 5)
    NN = np.arange(1, 101)
    e = np.array([float((cj.chebfun(lambda x: jnp.abs(x) ** 5, n=int(N)) - exact).norm(2))
                  for N in NN])
    theory = NN.astype(float) ** (-5.0)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.loglog(NN, e, '-', color=BLUE, linewidth=1.0)
    ax1.loglog(NN, theory, '--', color=RED, linewidth=1.0)
    ax1.text(5.5, 3e-7, r'$N^{-5}$', color=RED, fontsize=13)
    ax1.set_position(S2AT)
    _style_sub(ax1, xlim=(1, 100), ylim=(1e-10, 10),
               xticks=[1, 10, 100], yticks=[1e-10, 1e-5, 1e0],
               title='loglog scale', grid=True, which='both')
    ax2.semilogy(NN, e, '-', color=BLUE, linewidth=1.0)
    ax2.semilogy(NN, theory, '--', color=RED, linewidth=1.0)
    ax2.set_position(S2BT)
    _style_sub(ax2, xlim=(0, 100), ylim=(1e-10, 10),
               xticks=[0, 50, 100], yticks=[1e-10, 1e-5, 1e0],
               title='semilog scale', grid=True)
    _save(fig, 7)
except Exception as e:
    print(f"  guide04_07 FAILED: {e}")

# Fig 8:  Runge 1/(1+25x^2) convergence.
try:
    print("Fig 8: Runge convergence")
    exact = cj.chebfun(lambda x: 1.0 / (1 + 25 * x ** 2))
    NN = np.arange(1, 101)
    e = np.array([float((cj.chebfun(lambda x: 1.0 / (1 + 25 * x ** 2), n=int(N)) - exact).norm(2))
                  for N in NN])
    c = 1.0 / 5 + np.sqrt(1 + 1.0 / 25)
    theory = c ** (-NN.astype(float))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.loglog(NN, e, '-', color=BLUE, linewidth=1.0)
    ax1.loglog(NN, theory, '--', color=RED, linewidth=1.0)
    ax1.set_position(S2AT)
    _style_sub(ax1, xlim=(1, 100), ylim=(1e-10, 10),
               xticks=[1, 10, 100], yticks=[1e-10, 1e-5, 1e0],
               title='loglog scale', grid=True, which='both')
    ax2.semilogy(NN, e, '-', color=BLUE, linewidth=1.0)
    ax2.semilogy(NN, theory, '--', color=RED, linewidth=1.0)
    ax2.text(44, 3e-4, r'$C^{-N}$', color=RED, fontsize=13)
    ax2.set_position(S2BT)
    _style_sub(ax2, xlim=(0, 100), ylim=(1e-10, 10),
               xticks=[0, 50, 100], yticks=[1e-10, 1e-5, 1e0],
               title='semilog scale', grid=True)
    _save(fig, 8)
except Exception as e:
    print(f"  guide04_08 FAILED: {e}")


# ==========================================================================
# Section 4.5  Five theorems
# ==========================================================================

# Fig 9:  subplot(2,1,k), cheb.gallery('sinefun1'/'sinefun2'), ylim([0 3.5])
#   sinefun1 = 1.75 + sin(50x);  sinefun2 = (1.75 + sin(50x))^1.0001
try:
    print("Fig 9: sinefun1, sinefun2")
    f1 = cj.chebfun(lambda x: 1.75 + jnp.sin(50 * x))
    f2 = cj.chebfun(lambda x: (1.75 + jnp.sin(50 * x)) ** 1.0001)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    for ax, f, lbl, posn in [(ax1, f1, 'sinefun1', V1), (ax2, f2, 'sinefun2', V2)]:
        xs = np.linspace(-1, 1, 4000)
        ax.plot(xs, np.asarray(f(jnp.array(xs))), '-', color=BLUE, linewidth=1.0)
        ax.set_position(posn)
        _style_sub(ax, xlim=(-1, 1), ylim=(0, 3.5), xticks=UNIT_X,
                   yticks=[0, 2], title=f'{lbl}, length = {len(f)}')
    _save(fig, 9)
except Exception as e:
    print(f"  guide04_09 FAILED: {e}")


# ==========================================================================
# Section 4.6  Best approximations and the minimax command
# ==========================================================================

_sqrt = lambda x: jnp.sqrt(jnp.abs(x - 3.0))
_sqrt_np = lambda x: np.sqrt(np.abs(x - 3.0))
_mm = minimax(_sqrt, 20, domain=(0.0, 4.0))
_p = cj.Chebfun.from_coeffs(jnp.array(_mm.coeffs), domain=Domain([0.0, 4.0]))
_err = float(_mm.err)
_pinterp = cj.chebfun(_sqrt, domain=[0, 4], n=21)

# Fig 10:  plot(f,'b',p,'r'), grid on   (f = sqrt(|x-3|), p = minimax(f,20))
try:
    print("Fig 10: sqrt(|x-3|) and degree-20 minimax")
    xx = np.linspace(0, 4, 3000)
    fig, ax = plt.subplots()
    ax.plot(xx, _sqrt_np(xx), '-', color=BLUE, linewidth=1.0)
    ax.plot(xx, np.asarray(_p(jnp.array(xx))), '-', color=RED, linewidth=1.0)
    _finish_single(fig, ax, 10, xlim=(0, 4), ylim=(0, 2),
                   xticks=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
                   yticks=[0, 0.5, 1, 1.5, 2], grid=True)
except Exception as e:
    print(f"  guide04_10 FAILED: {e}")

# Fig 11:  plot(f-p,'m'); plot ±err dashed black; ylim(3*err*[-1,1])
try:
    print("Fig 11: error curve f - p (equioscillation)")
    xx = np.linspace(0, 4, 3000)
    ec = _sqrt_np(xx) - np.asarray(_p(jnp.array(xx)))
    fig, ax = plt.subplots()
    ax.plot(xx, ec, '-', color=MAG, linewidth=1.0)
    ax.plot([0, 4], [_err, _err], '--', color=BLACK, linewidth=0.8)
    ax.plot([0, 4], [-_err, -_err], '--', color=BLACK, linewidth=0.8)
    _finish_single(fig, ax, 11, xlim=(0, 4), ylim=(-3 * _err, 3 * _err),
                   xticks=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
                   yticks=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
except Exception as e:
    print(f"  guide04_11 FAILED: {e}")

# Fig 12:  pinterp = chebfun(f,[0,4],21); plot(f-pinterp,'b')  (over fig 11)
try:
    print("Fig 12: interpolant error vs best-approx error")
    xx = np.linspace(0, 4, 3000)
    ec = _sqrt_np(xx) - np.asarray(_p(jnp.array(xx)))
    ei = _sqrt_np(xx) - np.asarray(_pinterp(jnp.array(xx)))
    fig, ax = plt.subplots()
    ax.plot(xx, ec, '-', color=MAG, linewidth=1.0)
    ax.plot([0, 4], [_err, _err], '--', color=BLACK, linewidth=0.8)
    ax.plot([0, 4], [-_err, -_err], '--', color=BLACK, linewidth=0.8)
    ax.plot(xx, ei, '-', color=BLUE, linewidth=1.0)
    _finish_single(fig, ax, 12, xlim=(0, 4), ylim=(-3 * _err, 3 * _err),
                   xticks=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
                   yticks=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
except Exception as e:
    print(f"  guide04_12 FAILED: {e}")

# Fig 13:  CF(5,5) of exp(x); plot(f-r,'c'); ±err dashed; ylim(2e-13*[-1,1])
#   `cf` is a library gap -> NumPy port cf_rational above.
try:
    print("Fig 13: CF (5,5) approximation of exp(x)")
    fe = cj.chebfun(lambda x: jnp.exp(x))
    a = np.asarray(fe.coeffs, dtype=float); M = len(a) - 1
    vs = float(np.max(np.abs(np.asarray(fe(jnp.array(np.linspace(-1, 1, 400)))))))
    r_ref, _, s = cf_rational(a, 5, 5, M, vs)
    xx = np.linspace(-1, 1, 3000)
    ec = np.exp(xx) - r_ref(xx)
    fig, ax = plt.subplots()
    ax.plot(xx, ec, '-', color=CYAN, linewidth=1.2)
    ax.plot([-1, 1], [s, s], '--', color=BLACK, linewidth=0.8)
    ax.plot([-1, 1], [-s, -s], '--', color=BLACK, linewidth=0.8)
    _finish_single(fig, ax, 13, xlim=(-1, 1), ylim=(-2e-13, 2e-13),
                   xticks=UNIT_X, yticks=[-2e-13, -1e-13, 0, 1e-13, 2e-13])
except Exception as e:
    print(f"  guide04_13 FAILED: {e}")

# Fig 14:  CF(5,5) of |x-.3| with 4th arg 300; plot(f-p/q,'c'); ±lam dashed
try:
    print("Fig 14: CF (5,5) approximation of |x-0.3|")
    fa = cj.chebfun(lambda x: jnp.abs(x - 0.3), n=301)
    a = np.asarray(fa.coeffs, dtype=float); M = 300
    vs = float(np.max(np.abs(np.asarray(fa(jnp.array(np.linspace(-1, 1, 500)))))))
    r_ref, _, lam = cf_rational(a, 5, 5, M, vs)
    xx = np.linspace(-1, 1, 6000)
    ec = np.abs(xx - 0.3) - r_ref(xx)
    fig, ax = plt.subplots()
    ax.plot(xx, ec, '-', color=CYAN, linewidth=1.2)
    ax.plot([-1, 1], [lam, lam], '--', color=BLACK, linewidth=0.8)
    ax.plot([-1, 1], [-lam, -lam], '--', color=BLACK, linewidth=0.8)
    ymax = 1.15 * float(np.max(np.abs(ec)))
    _finish_single(fig, ax, 14, xlim=(-1, 1), ylim=(-ymax, ymax),
                   xticks=UNIT_X, yticks=[-5e-3, 0, 5e-3])
except Exception as e:
    print(f"  guide04_14 FAILED: {e}")


# ==========================================================================
# Section 4.7  The Runge phenomenon
# ==========================================================================

_ftanh10 = lambda x: jnp.tanh(10 * x)
_ftanh10_np = lambda x: np.tanh(10 * x)

# Fig 15:  equispaced degree-9 interpolation of tanh(10x)
try:
    print("Fig 15: equispaced interpolation, 10 points")
    s = np.linspace(-1, 1, 10)
    p = cj.Chebfun.interp1(jnp.array(s), _ftanh10(jnp.array(s)))
    xx = np.linspace(-1, 1, 2000)
    fig, ax = plt.subplots()
    ax.plot(xx, _ftanh10_np(xx), '-', color=BLUE, linewidth=1.0)
    ax.plot(xx, np.asarray(p(jnp.array(xx))), '-', color=RED, linewidth=1.0)
    ax.plot(s, np.asarray(p(jnp.array(s))), '.', color=RED, markersize=8)
    _finish_single(fig, ax, 15, xlim=(-1, 1), ylim=(-1.5, 1.5), xticks=UNIT_X,
                   yticks=[-1.5, -1, -0.5, 0, 0.5, 1, 1.5], grid=True)
except Exception as e:
    print(f"  guide04_15 FAILED: {e}")

# Fig 16:  equispaced degree-19 interpolation of tanh(10x)
try:
    print("Fig 16: equispaced interpolation, 20 points")
    s = np.linspace(-1, 1, 20)
    p = cj.Chebfun.interp1(jnp.array(s), _ftanh10(jnp.array(s)))
    xx = np.linspace(-1, 1, 2000)
    fig, ax = plt.subplots()
    ax.plot(xx, _ftanh10_np(xx), '-', color=BLUE, linewidth=1.0)
    ax.plot(xx, np.asarray(p(jnp.array(xx))), '-', color=RED, linewidth=1.0)
    ax.plot(s, np.asarray(p(jnp.array(s))), '.', color=RED, markersize=8)
    _finish_single(fig, ax, 16, xlim=(-1, 1), ylim=(-150, 150), xticks=UNIT_X,
                   yticks=[-150, -100, -50, 0, 50, 100, 150], grid=True)
except Exception as e:
    print(f"  guide04_16 FAILED: {e}")

# Fig 17:  semilogy(lebesgue(s))  with s = 20 equispaced points
try:
    print("Fig 17: Lebesgue function, 20 equispaced points")
    s = np.linspace(-1, 1, 20)
    t, lam = lebesgue_function(jnp.array(s))
    fig, ax = plt.subplots()
    ax.semilogy(np.asarray(t), np.asarray(lam), '-', color=BLUE, linewidth=1.0)
    _finish_single(fig, ax, 17, xlim=(-1, 1), ylim=(1, 1e4), xticks=UNIT_X,
                   yticks=[1e0, 1e1, 1e2, 1e3, 1e4])
except Exception as e:
    print(f"  guide04_17 FAILED: {e}")

# Fig 18:  semilogy(lebesgue(linspace(-1,1,40)))
try:
    print("Fig 18: Lebesgue function, 40 equispaced points")
    s = np.linspace(-1, 1, 40)
    t, lam = lebesgue_function(jnp.array(s))
    fig, ax = plt.subplots()
    ax.semilogy(np.asarray(t), np.asarray(lam), '-', color=BLUE, linewidth=1.0)
    _finish_single(fig, ax, 18, xlim=(-1, 1), ylim=(1, 1e10), xticks=UNIT_X,
                   yticks=[1e0, 1e5, 1e10])
except Exception as e:
    print(f"  guide04_18 FAILED: {e}")


# ==========================================================================
# Section 4.8  Rational approximations
# ==========================================================================

_ftest = cj.chebfun(lambda x: jnp.tanh(jnp.pi * x / 2) + x / 20, domain=[-10, 10])
_ftest_np = lambda x: np.tanh(np.pi * x / 2) + x / 20

# Fig 19:  plot(f)  with f = tanh(pi*x/2) + x/20 on [-10,10]
try:
    print("Fig 19: tanh(pi*x/2) + x/20 on [-10,10]")
    xx = np.linspace(-10, 10, 2000)
    fig, ax = plt.subplots()
    ax.plot(xx, np.asarray(_ftest(jnp.array(xx))), '-', color=BLUE, linewidth=1.0)
    _finish_single(fig, ax, 19, xlim=(-10, 10), ylim=(-1.5, 1.5),
                   xticks=[-10, -5, 0, 5, 10],
                   yticks=[-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
except Exception as e:
    print(f"  guide04_19 FAILED: {e}")

# Fig 20:  [p,q]=chebpade(f,40,4); r=p/q; plot(f-r,'r')
try:
    print("Fig 20: Chebyshev-Pade (40,4) error")
    _, _, rh = chebpade(_ftest, 40, 4)
    xx = np.linspace(-10, 10, 2000)
    rv = np.array([float(rh(float(t / 10.0))) for t in xx])
    err = np.asarray(_ftest(jnp.array(xx))) - rv
    fig, ax = plt.subplots()
    ax.plot(xx, err, '-', color=RED, linewidth=1.0)
    ymax = 1.1 * float(np.max(np.abs(err)))
    _finish_single(fig, ax, 20, xlim=(-10, 10), ylim=(-ymax, ymax),
                   xticks=[-10, -5, 0, 5, 10],
                   yticks=[-4e-9, -2e-9, 0, 2e-9, 4e-9])
except Exception as e:
    print(f"  guide04_20 FAILED: {e}")

# Fig 21:  [p,q]=ratinterp(f,40,4); r=p/q; plot(f-r,'m')
try:
    print("Fig 21: rational interpolation (40,4) error")
    ri = ratinterp(_ftest, 40, 4, domain=(-10.0, 10.0))
    rh = ri[0]
    xx = np.linspace(-10, 10, 2000)
    rv = np.array([float(rh(float(t))) for t in xx])
    err = np.asarray(_ftest(jnp.array(xx))) - rv
    fig, ax = plt.subplots()
    ax.plot(xx, err, '-', color=MAG, linewidth=1.0)
    ymax = 1.1 * float(np.max(np.abs(err)))
    _finish_single(fig, ax, 21, xlim=(-10, 10), ylim=(-ymax, ymax),
                   xticks=[-10, -5, 0, 5, 10],
                   yticks=[-4e-7, -2e-7, 0, 2e-7, 4e-7])
except Exception as e:
    print(f"  guide04_21 FAILED: {e}")

# Fig 22:  [p,q]=cf(f,40,4); r=p/q; plot(f-r,'c')   (CF: NumPy port)
try:
    print("Fig 22: Caratheodory-Fejer (40,4) error")
    a = np.asarray(_ftest.coeffs, dtype=float); M = len(a) - 1
    vs = float(np.max(np.abs(np.asarray(_ftest(jnp.array(np.linspace(-10, 10, 500)))))))
    r_ref, _, s = cf_rational(a, 40, 4, M, vs)
    xx = np.linspace(-10, 10, 3000)
    err = np.asarray(_ftest(jnp.array(xx))) - r_ref(xx / 10.0)
    fig, ax = plt.subplots()
    ax.plot(xx, err, '-', color=CYAN, linewidth=1.0)
    ymax = 1.05 * float(np.max(np.abs(err)))
    _finish_single(fig, ax, 22, xlim=(-10, 10), ylim=(-ymax, ymax),
                   xticks=[-10, -5, 0, 5, 10], yticks=[-1e-10, 0, 1e-10])
except Exception as e:
    print(f"  guide04_22 FAILED: {e}")

# --------------------------------------------------------------------------
# Fig 23:  extra reference committed to the repo — a matplotlib-default render
#   of the rational-interpolation error obtained WITHOUT passing the physical
#   domain to ratinterp (so the type-(40,4) fit is built on [-1,1] and blows
#   up when extrapolated to [-10,10]).  Reproduced here to match the committed
#   reference; this figure does not appear on chebfun.org.
# --------------------------------------------------------------------------
try:
    print("Fig 23: ratinterp extrapolation blow-up (matplotlib default)")
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        ri = ratinterp(_ftest, 40, 4)   # no domain: fit on [-1,1]
        rh = ri[0]
        xx = np.linspace(-10, 10, 2000)
        rv = np.array([float(rh(float(t / 10.0))) for t in xx])
        err = np.asarray(_ftest(jnp.array(xx))) - rv
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xx, err, color=MAG, linewidth=1.0)
        ax.grid(True, alpha=0.3)
        ax.set_title('Rational interpolation error (ratinterp)')
        path = os.path.join(OUT_DIR, 'guide04_23.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    print("  guide04_23.png saved")
except Exception as e:
    print(f"  guide04_23 FAILED: {e}")

print("\nGuide 04 plot generation complete.")
