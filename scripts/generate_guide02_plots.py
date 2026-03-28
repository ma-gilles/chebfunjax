"""Generate all plots for Guide Chapter 2: Integration and Differentiation.

This script reproduces every figure from Chebfun Guide Chapter 2 using
chebfunjax.  Each plot is saved as docs/images/guide/guide02_NN.png.
"""

import matplotlib
matplotlib.use('Agg')

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import scipy.special as sp
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, contour as contour_2d

chebfun_style()

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT_DIR, exist_ok=True)

plot_idx = 0

# --------------------------------------------------------------------------
# Plot 1 (guide02_01): |J_0(x)| on [0, 20]
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    g = cj.chebfun(lambda t: sp.jv(0, np.asarray(t, dtype=np.float64)), domain=[0, 20])
    g_abs = g.abs()
    fig, ax = cj.plot(g_abs)
    ax.set_ylim([0, 1.1])
    fig.savefig(os.path.join(OUT_DIR, f'guide02_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide02_{plot_idx:02d}.png saved")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"guide02_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 2 (guide02_02): min(sech(3*sin(10*x)), sin(9*x)) on [-1,1]
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    f = cj.chebfun(lambda x: 1.0 / jnp.cosh(3.0 * jnp.sin(10.0 * x)))
    g = cj.chebfun(lambda x: jnp.sin(9.0 * x))
    # min(f, g) - need piecewise construction
    diff_fg = f - g
    crossings = diff_fg.roots()
    breaks = [-1.0] + sorted([float(r) for r in crossings]) + [1.0]
    # Remove near-duplicates
    clean = [breaks[0]]
    for b in breaks[1:]:
        if b - clean[-1] > 1e-12:
            clean.append(b)
    breaks = clean

    from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
    from chebfunjax.domain import Domain
    piece_list = []
    for i in range(len(breaks) - 1):
        mid = 0.5 * (breaks[i] + breaks[i+1])
        fval = float(f(jnp.float64(mid)))
        gval = float(g(jnp.float64(mid)))
        if fval <= gval:
            piece_list.append(_Piece.from_function(lambda x, _f=f: _f(x), breaks[i], breaks[i+1]))
        else:
            piece_list.append(_Piece.from_function(lambda x, _g=g: _g(x), breaks[i], breaks[i+1]))
    h = Chebfun(funs=piece_list, domain=Domain(tuple(breaks)))
    fig, ax = cj.plot(h)
    fig.savefig(os.path.join(OUT_DIR, f'guide02_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide02_{plot_idx:02d}.png saved")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"guide02_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 3 (guide02_03): Kahaner's F21F function with three spikes
# --------------------------------------------------------------------------
try:
    plot_idx += 1

    def ff(x):
        return (1.0 / jnp.cosh(10.0 * (x - 0.2)))**2 + \
               (1.0 / jnp.cosh(100.0 * (x - 0.4)))**4 + \
               (1.0 / jnp.cosh(1000.0 * (x - 0.6)))**6

    f = cj.chebfun(ff, domain=[0, 1])
    fig, ax = cj.plot(f)
    fig.savefig(os.path.join(OUT_DIR, f'guide02_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide02_{plot_idx:02d}.png saved")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"guide02_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 4 (guide02_04): exp(-1/sin(10*x)^2) on [-1,1]
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    f = cj.chebfun(lambda x: jnp.exp(-1.0 / jnp.sin(10.0 * x)**2))
    fig, ax = cj.plot(f)
    fig.savefig(os.path.join(OUT_DIR, f'guide02_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide02_{plot_idx:02d}.png saved")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"guide02_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 5 (guide02_05): cumsum of erf integrand, raw (F(a)=0)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    t = cj.chebfun(lambda t: t, domain=[-5, 5])
    f = cj.chebfun(lambda t: 2.0 * jnp.exp(-t**2) / jnp.sqrt(jnp.pi), domain=[-5, 5])
    fint = f.cumsum()
    fig, ax = cj.plot(fint, color='m')
    ax.set_ylim([-0.2, 2.2])
    ax.grid(True)
    fig.savefig(os.path.join(OUT_DIR, f'guide02_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide02_{plot_idx:02d}.png saved")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"guide02_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 6 (guide02_06): cumsum shifted so F(0)=0 (our erf function)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    f = cj.chebfun(lambda t: 2.0 * jnp.exp(-t**2) / jnp.sqrt(jnp.pi), domain=[-5, 5])
    fint = f.cumsum()
    shift = float(fint(jnp.float64(0.0)))
    fint_shifted = fint - shift
    fig, ax = cj.plot(fint_shifted, color='m')
    ax.set_ylim([-1.2, 1.2])
    ax.grid(True)
    fig.savefig(os.path.join(OUT_DIR, f'guide02_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide02_{plot_idx:02d}.png saved")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"guide02_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 7 (guide02_07): oscillatory step function and its integral
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    # x * sign(sin(x^2)) on [0, 6], piecewise
    # We need the roots of sin(x^2) on [0, 6]: x = sqrt(k*pi) for k=0,1,...
    k_vals = np.arange(0, 12)  # sqrt(11*pi) ~ 5.88 < 6
    breaks_inner = np.sqrt(k_vals * np.pi)
    breaks_inner = breaks_inner[(breaks_inner >= 0) & (breaks_inner <= 6)]
    breaks = [0.0] + list(breaks_inner[breaks_inner > 0]) + [6.0]
    # Remove duplicates and sort
    breaks = sorted(set(breaks))

    from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
    from chebfunjax.domain import Domain
    piece_list = []
    for i in range(len(breaks) - 1):
        mid = 0.5 * (breaks[i] + breaks[i+1])
        s = np.sign(np.sin(mid**2))
        piece_list.append(_Piece.from_function(
            lambda x, _s=s: x * _s, breaks[i], breaks[i+1]
        ))
    f_pw = Chebfun(funs=piece_list, domain=Domain(tuple(breaks)))
    g_pw = f_pw.cumsum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    cj.plot(f_pw, ax=ax1)
    cj.plot(g_pw, ax=ax2, color='m')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide02_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide02_{plot_idx:02d}.png saved")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"guide02_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 8 (guide02_08): Li(x) vs pi(x) (prime counting function)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    mu = 1.45136923488338105   # Soldner's constant
    xmax = 400
    Li = cj.chebfun(lambda x: 1.0 / jnp.log(x), domain=[mu, xmax]).cumsum()

    # Sieve of Eratosthenes for primes up to xmax
    def primes_up_to(n):
        sieve = np.ones(n+1, dtype=bool)
        sieve[:2] = False
        for i in range(2, int(n**0.5)+1):
            if sieve[i]:
                sieve[i*i::i] = False
        return np.where(sieve)[0]

    p = primes_up_to(xmax)

    fig, ax = plt.subplots(figsize=(8, 5))
    cj.plot(Li, ax=ax, color='m')
    ax.plot(p, np.arange(1, len(p)+1), '.k', markersize=2)
    fig.savefig(os.path.join(OUT_DIR, f'guide02_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide02_{plot_idx:02d}.png saved")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"guide02_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 9 (guide02_09): cos(pi*x) and its derivative -pi*sin(pi*x) on [0,20]
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    f = cj.chebfun(lambda x: jnp.cos(jnp.pi * x), domain=[0, 20])
    fprime = f.diff()
    fig, ax = plt.subplots(figsize=(8, 4))
    cj.plot_1d(f, ax=ax)
    cj.plot_1d(fprime, ax=ax, color=cj.plotting.CHEBFUN_RED)
    fig.savefig(os.path.join(OUT_DIR, f'guide02_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide02_{plot_idx:02d}.png saved")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"guide02_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 10 (guide02_10): piecewise function f = {x^2, 1, 4-x, 4/x} on [0,4]
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
    from chebfunjax.domain import Domain

    piece_list = [
        _Piece.from_function(lambda x: x**2, 0, 1),
        _Piece.from_function(lambda x: jnp.ones_like(x), 1, 2),
        _Piece.from_function(lambda x: 4.0 - x, 2, 3),
        _Piece.from_function(lambda x: 4.0 / x, 3, 4),
    ]
    f_pw = Chebfun(funs=piece_list, domain=Domain((0.0, 1.0, 2.0, 3.0, 4.0)))
    fig, ax = cj.plot(f_pw)
    fig.savefig(os.path.join(OUT_DIR, f'guide02_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide02_{plot_idx:02d}.png saved")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"guide02_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 11 (guide02_11): derivative of the piecewise function
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    fprime_pw = f_pw.diff()
    fig, ax = cj.plot(fprime_pw, color='r')
    ax.set_ylim([-2, 3])
    fig.savefig(os.path.join(OUT_DIR, f'guide02_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide02_{plot_idx:02d}.png saved")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"guide02_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 12 (guide02_12): 4th derivative of 1/(1+x^2)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    f = cj.chebfun(lambda x: 1.0 / (1.0 + x**2))
    g = f.diff(4)
    fig, ax = cj.plot(g)
    fig.savefig(os.path.join(OUT_DIR, f'guide02_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide02_{plot_idx:02d}.png saved")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"guide02_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 13 (guide02_13): 2D contour plot of sin(5*(theta - r))*sin(x)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    r_fn = lambda x, y: jnp.sqrt(x**2 + y**2)
    theta_fn = lambda x, y: jnp.arctan2(y, x)
    f2d = lambda x, y: jnp.sin(5.0 * (theta_fn(x, y) - r_fn(x, y))) * jnp.sin(x)

    f2_obj = cj.chebfun2(f2d, domain=(-2, 2, 0.5, 2.5))
    fig, ax = contour_2d(f2_obj, title='', filled=False)
    fig.savefig(os.path.join(OUT_DIR, f'guide02_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide02_{plot_idx:02d}.png saved")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"guide02_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 14 (guide02_14): same 2D contour via Chebfun2
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    f2 = cj.chebfun2(f2d, domain=(-2, 2, 0.5, 2.5))
    fig, ax = contour_2d(f2, title='Chebfun2 contour')
    fig.savefig(os.path.join(OUT_DIR, f'guide02_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide02_{plot_idx:02d}.png saved")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"guide02_{plot_idx:02d}.png FAILED: {e}")

print(f"\nGuide 02: generated {plot_idx} plots total.")
