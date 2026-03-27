"""Generate all plots for Guide Chapter 1.

Produces one PNG for every code block in the chapter that creates a figure.
Files are saved as docs/images/guide/guide01_NN.png where NN is the sequential
plot number matching the order in the original Chebfun Guide Chapter 1.
"""

import matplotlib
matplotlib.use('Agg')

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import scipy.special as sp
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT_DIR, exist_ok=True)

plot_idx = 0


def save(fig, idx):
    path = os.path.join(OUT_DIR, f'guide01_{idx:02d}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  guide01_{idx:02d}.png saved")


# --------------------------------------------------------------------------
# Plot 1: cos(20x) on [-1,1]   (Section 1.2)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: cos(20x)")
    f = cj.chebfun(lambda x: jnp.cos(20 * x))
    fig, ax = f.plot()
    ax.set_ylim(-1.2, 1.2)
    save(fig, plot_idx)
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 2: cos(20x) with Chebyshev points shown ('.-')   (Section 1.2)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: cos(20x) with Chebyshev points")
    f = cj.chebfun(lambda x: jnp.cos(20 * x))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    # Plot the smooth curve
    xs = np.linspace(-1, 1, 600)
    ys = np.array(f(jnp.array(xs)))
    ax.plot(xs, ys, '-', color='#2855A0', linewidth=1.8)
    # Plot Chebyshev points
    n = len(f)
    cheb_pts = -np.cos(np.pi * np.arange(n) / (n - 1))
    cheb_vals = np.array(f(jnp.array(cheb_pts)))
    ax.plot(cheb_pts, cheb_vals, '.', color='#2855A0', markersize=6)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('x')
    fig.set_facecolor('white')
    fig.tight_layout()
    save(fig, plot_idx)
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 3: Bessel J0 on [0,100]   (Section 1.2)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: Bessel J0 on [0,100]")
    g = cj.chebfun(lambda t: jnp.array(sp.j0(np.asarray(t))), domain=[0, 100])
    fig, ax = g.plot()
    ax.set_ylim(-0.5, 1)
    save(fig, plot_idx)
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 4: Runge function 1/(1+25x^2)   (Section 1.2)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: Runge function")
    x = cj.chebfun(lambda x: x)
    f = 1 / (1 + 25 * x**2)
    fig, ax = f.plot()
    save(fig, plot_idx)
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 5: Piecewise function {x^2, 1, 4-x} on [-1,1,2,4]   (Section 1.4)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: Piecewise x^2, 1, 4-x")
    # Build piecewise: x^2 on [-1,1], 1 on [1,2], 4-x on [2,4]
    from chebfunjax.chebfun1d.chebfun import Chebfun
    from chebfunjax.domain import Domain
    f1 = cj.chebfun(lambda x: x**2, domain=[-1, 1])
    f2 = cj.chebfun(lambda x: jnp.ones_like(x), domain=[1, 2])
    f3 = cj.chebfun(lambda x: 4.0 - x, domain=[2, 4])
    funs = f1.funs + f2.funs + f3.funs
    dom = Domain((-1.0, 1.0, 2.0, 4.0))
    f = Chebfun(funs=funs, domain=dom)
    fig, ax = f.plot()
    save(fig, plot_idx)
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 6: 1/(1+f) where f is the piecewise function, in red   (Section 1.4)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: 1/(1+f) piecewise, red")
    # f is still the piecewise function from plot 5
    g = 1 / (1 + f)
    fig, ax = g.plot(color='r')
    save(fig, plot_idx)
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 7: abs(exp(x)*sin(8x))   (Section 1.4)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: abs(exp(x)*sin(8x))")
    x = cj.chebfun(lambda x: x)
    f = (x.exp() * (8 * x).sin()).abs()
    fig, ax = f.plot()
    save(fig, plot_idx)
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 8: max(sin(20x), exp(x-1))   (Section 1.4)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: max(sin(20x), exp(x-1))")
    x = cj.chebfun(lambda x: x)
    f_sin = (20 * x).sin()
    g_exp = (x - 1).exp()
    # Find crossover points where sin(20x) = exp(x-1)
    diff_fg = f_sin - g_exp
    r = diff_fg.roots()
    bps = np.sort(np.concatenate([[-1.0], np.asarray(r), [1.0]]))
    # Build piecewise max
    h = cj.chebfun(
        lambda t: jnp.maximum(jnp.sin(20 * t), jnp.exp(t - 1)),
        domain=list(bps)
    )
    fig, ax = h.plot()
    ax.set_ylim(0, 1.2)
    save(fig, plot_idx)
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 9: exp(-x^2/16)*(1+0.2*cos(10x)) on [-inf,inf]   (Section 1.5)
# NOTE: Infinite intervals not yet supported. We approximate on [-20,20].
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: Gaussian-modulated cosine (finite approx)")
    f = cj.chebfun(
        lambda x: jnp.exp(-x**2 / 16) * (1 + 0.2 * jnp.cos(10 * x)),
        domain=[-20, 20]
    )
    fig, ax = f.plot()
    save(fig, plot_idx)
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 10: (1/pi)/sqrt(1-x^2)   (Section 1.5)
# NOTE: Endpoint singularities ('exps') not yet supported.
# We plot a smooth approximation avoiding endpoints.
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: Arcsine distribution (smooth approx)")
    eps = 1e-6
    f = cj.chebfun(
        lambda x: (1 / jnp.pi) / jnp.sqrt(1 - x**2 + eps),
        domain=[-1 + eps, 1 - eps]
    )
    fig, ax = f.plot()
    save(fig, plot_idx)
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 11: Periodic function (Chebyshev representation)   (Section 1.6)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: Periodic function (Chebyshev)")
    ff = lambda t: jnp.sin(t) + jnp.cos(2 * t) - jnp.cos(t) / 3 + jnp.cos(100 * t) / 6
    f = cj.chebfun(ff, domain=[-jnp.pi, jnp.pi])
    fig, ax = f.plot()
    save(fig, plot_idx)
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 12: Same periodic function (same since we don't have 'trig' mode)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: Periodic function (magenta)")
    ff = lambda t: jnp.sin(t) + jnp.cos(2 * t) - jnp.cos(t) / 3 + jnp.cos(100 * t) / 6
    f2 = cj.chebfun(ff, domain=[-jnp.pi, jnp.pi])
    fig, ax = f2.plot(color='m')
    save(fig, plot_idx)
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 13: Airy function on [-40,40]   (Section 1.9)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: Airy function")
    f = cj.chebfun(lambda x: jnp.array(sp.airy(np.asarray(x))[0]), domain=[-40, 40])
    fig, ax = f.plot()
    ax.set_ylim(-0.8, 0.8)
    ax.set_title('Airy function')
    save(fig, plot_idx)
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 14: Daubechies scaling function approximation   (Section 1.9)
# NOTE: The Daubechies wavelet is not in the gallery. We show a placeholder.
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: Daubechies (approximation)")
    # Approximate the Daubechies phi_2 scaling function by iterated refinement
    # Use the known filter coefficients for D4 wavelet
    h = np.array([
        (1 + np.sqrt(3)) / (4 * np.sqrt(2)),
        (3 + np.sqrt(3)) / (4 * np.sqrt(2)),
        (3 - np.sqrt(3)) / (4 * np.sqrt(2)),
        (1 - np.sqrt(3)) / (4 * np.sqrt(2)),
    ])
    # Cascade algorithm: start with hat function on [0,3]
    N = 12  # number of iterations
    x_pts = np.linspace(0, 3, 3 * 2**N + 1)
    phi = np.where((x_pts >= 0) & (x_pts <= 3), 1.0, 0.0)
    for _ in range(N):
        phi_new = np.zeros_like(phi)
        for k in range(len(h)):
            # phi_new(x) += sqrt(2) * h[k] * phi(2x - k)
            shifted = np.interp(2 * x_pts - k, x_pts, phi, left=0, right=0)
            phi_new += np.sqrt(2) * h[k] * shifted
        phi = phi_new
    f = cj.chebfun(lambda x: jnp.interp(x, jnp.array(x_pts), jnp.array(phi)),
                    domain=[0, 3])
    fig, ax = f.plot()
    ax.set_ylim(-0.5, 1.5)
    ax.set_title('Daubechies scaling function')
    save(fig, plot_idx)
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 15: Zigzag polynomial   (Section 1.9)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: Zigzag polynomial")
    from chebfunjax.utils.gallery import gallery
    f = gallery('zigzag')
    fig, ax = f.plot()
    ax.set_title('zigzag')
    save(fig, plot_idx)
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")


print("\nGuide 01 plot generation complete.")
