"""Generate all plots for Guide Chapter 11: Periodic Chebfuns."""

import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import warnings
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, CHEBFUN_BLUE, CHEBFUN_RED
from chebfunjax.tech.trigtech import Trigtech, trig_vals2coeffs, trig_coeffs2vals, trigpts

chebfun_style()

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT_DIR, exist_ok=True)

plot_idx = 0


def save(fig, hint=""):
    global plot_idx
    plot_idx += 1
    path = os.path.join(OUT_DIR, f'guide11_{plot_idx:02d}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  guide11_{plot_idx:02d}.png saved  ({hint})")


# --------------------------------------------------------------------------
# Plot 1: f = tanh(3*sin(t)) - sin(t+1/2) on [-pi,pi]   (Section 11.1)
# --------------------------------------------------------------------------
try:
    # Map [-pi,pi] -> [-1,1]: t = pi*s
    f = Trigtech.from_function(
        lambda s: jnp.tanh(3 * jnp.sin(jnp.pi * s)) - jnp.sin(jnp.pi * s + 0.5),
    )
    xs_ref = np.linspace(-1, 1, 600)
    xs_phys = xs_ref * np.pi
    ys = np.array(f(jnp.array(xs_ref)))
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(xs_phys, ys, color=CHEBFUN_BLUE, linewidth=1.8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "tanh(3sin(t))-sin(t+1/2)")
except Exception as e:
    plot_idx += 1
    print(f"  guide11_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 2: projection vs interpolation error for |sin(t)|^3  (Section 11.2)
# --------------------------------------------------------------------------
try:
    uu = lambda s: jnp.abs(jnp.sin(jnp.pi * s))**3
    u_full = Trigtech.from_function(uu)

    # degree-5 projection (trunc) and interpolation
    q11 = Trigtech.from_function(uu, n=11)

    # For the "projection" we truncate the full coefficients to 11 modes
    c_full = np.array(u_full.coeffs)
    M_full = len(c_full) // 2
    trunc_c = np.zeros(11, dtype=np.complex128)
    for k in range(-5, 6):
        trunc_c[k + 5] = c_full[k + M_full]
    p11 = Trigtech(coeffs=jnp.array(trunc_c), is_real=True, ishappy=True)

    xs = np.linspace(-1, 1, 600)
    ys_u = np.array(u_full(jnp.array(xs)))
    ys_q = np.array(q11(jnp.array(xs)))
    ys_p = np.array(p11(jnp.array(xs)))

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(xs * np.pi, ys_p - ys_u, color=CHEBFUN_BLUE, linewidth=1.5, label='projection error')
    ax.plot(xs * np.pi, ys_q - ys_u, color=CHEBFUN_RED, linewidth=1.5, label='interpolation error')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "projection vs interpolation error")
except Exception as e:
    plot_idx += 1
    print(f"  guide11_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 3: f = tanh(cos(1+2g)^2)+g/3-0.5 with extrema and roots (Sec 11.3)
# --------------------------------------------------------------------------
try:
    g = Trigtech.from_function(lambda s: jnp.sin(jnp.pi * s))
    f_vals_fn = lambda s: (jnp.tanh(jnp.cos(1.0 + 2.0 * jnp.sin(jnp.pi * s))**2)
                           + jnp.sin(jnp.pi * s) / 3.0 - 0.5)
    f = Trigtech.from_function(f_vals_fn)

    xs = np.linspace(-1, 1, 600)
    xs_phys = xs * np.pi
    ys = np.array(f(jnp.array(xs)))

    # Find roots
    r = np.array(f.roots())

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(xs_phys, ys, color=CHEBFUN_BLUE, linewidth=1.8)
    if len(r) > 0:
        yr = np.array(f(jnp.array(r)))
        ax.plot(r * np.pi, yr, '.r', markersize=14)
    # Mark max and min
    ys_dense = np.array(f(jnp.array(xs)))
    imax = np.argmax(ys_dense)
    imin = np.argmin(ys_dense)
    ax.plot(xs_phys[imax], ys_dense[imax], 'ok', markersize=8)
    ax.plot(xs_phys[imin], ys_dense[imin], 'ok', markersize=8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "f with roots and extrema")
except Exception as e:
    plot_idx += 1
    print(f"  guide11_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 4: |f| (abs breaks trigonometric representation)  (Section 11.3)
# --------------------------------------------------------------------------
try:
    ys_abs = np.abs(ys)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(xs_phys, ys_abs, color=CHEBFUN_BLUE, linewidth=1.8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "|f| (abs of trigfun)")
except Exception as e:
    plot_idx += 1
    print(f"  guide11_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 5: Noisy function + Gaussian smoothing (circconv)  (Sec 11.5)
# --------------------------------------------------------------------------
try:
    np.random.seed(0)
    n_pts = 201
    tt = np.linspace(-np.pi, np.pi, n_pts, endpoint=False)
    ff_vals = np.exp(np.sin(tt)) + 0.05 * np.random.randn(n_pts)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(tt, ff_vals, color=CHEBFUN_BLUE, linewidth=0.8, alpha=0.7)

    # Gaussian convolution (smoothing)
    sigma = 0.1
    gaussian = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (tt / sigma)**2)
    # Circular convolution via FFT
    dt = 2 * np.pi / n_pts
    h_vals = np.real(np.fft.ifft(np.fft.fft(ff_vals) * np.fft.fft(gaussian))) * dt
    ax.plot(tt, h_vals, color=CHEBFUN_RED, linewidth=1.8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "noisy function + Gaussian smoothing")
except Exception as e:
    plot_idx += 1
    print(f"  guide11_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 6: plotcoeffs for exp(sin(t)) -- entire function  (Section 11.7)
# --------------------------------------------------------------------------
try:
    f_ent = Trigtech.from_function(lambda s: jnp.exp(jnp.sin(jnp.pi * s)))
    c = np.abs(np.array(f_ent.coeffs))
    c = np.maximum(c, 1e-18)
    ks = np.arange(len(c)) - len(c) // 2

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.semilogy(ks, c, '.', color=CHEBFUN_BLUE, markersize=4)
    ax.set_xlabel('Fourier mode $k$')
    ax.set_ylabel('$|c_k|$')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "plotcoeffs exp(sin(t))")
except Exception as e:
    plot_idx += 1
    print(f"  guide11_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 7: plotcoeffs for 1/(2-cos(t)) -- geometric decay  (Section 11.7)
# --------------------------------------------------------------------------
try:
    f_geo = Trigtech.from_function(lambda s: 1.0 / (2.0 - jnp.cos(jnp.pi * s)))
    c = np.abs(np.array(f_geo.coeffs))
    c = np.maximum(c, 1e-18)
    ks = np.arange(len(c)) - len(c) // 2

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.semilogy(ks, c, '.', color=CHEBFUN_BLUE, markersize=4)
    ax.set_xlabel('Fourier mode $k$')
    ax.set_ylabel('$|c_k|$')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "plotcoeffs 1/(2-cos(t))")
except Exception as e:
    plot_idx += 1
    print(f"  guide11_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 8: plotcoeffs for |sin(t)|^5 -- algebraic decay  (Section 11.7)
# --------------------------------------------------------------------------
try:
    f_alg = Trigtech.from_function(lambda s: jnp.abs(jnp.sin(jnp.pi * s))**5)
    c = np.abs(np.array(f_alg.coeffs))
    c = np.maximum(c, 1e-18)
    ks = np.arange(len(c)) - len(c) // 2

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.semilogy(ks, c, '.', color=CHEBFUN_BLUE, markersize=4)
    ax.set_xlabel('Fourier mode $k$')
    ax.set_ylabel('$|c_k|$')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "plotcoeffs |sin(t)|^5")
except Exception as e:
    plot_idx += 1
    print(f"  guide11_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 9: loglog plotcoeffs for |sin(t)|^5 with k^{-6} ref  (Sec 11.7)
# --------------------------------------------------------------------------
try:
    # Use only positive modes for loglog
    mid = len(c) // 2
    c_pos = c[mid + 1:]  # k = 1, 2, ...
    ks_pos = np.arange(1, len(c_pos) + 1)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.loglog(ks_pos, c_pos, '.', color=CHEBFUN_BLUE, markersize=4)
    # Reference line k^{-6}
    ks_ref = np.array([3, 300])
    ax.loglog(ks_ref, 3.0 * ks_ref.astype(float)**(-6), '--r', linewidth=1.2)
    ax.text(110, 4e-9, '$k^{-6}$', color='r', fontsize=11)
    ax.set_xlabel('Fourier mode $k$')
    ax.set_ylabel('$|c_k|$')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "loglog |sin(t)|^5 coeffs")
except Exception as e:
    plot_idx += 1
    print(f"  guide11_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 10: Gibbs phenomenon: square wave + truncated Fourier  (Sec 11.8)
# --------------------------------------------------------------------------
try:
    # Square wave via splitting
    sq_wave = lambda s: np.sign(np.sin(np.pi * s))
    n_sq = 201
    t_sq = np.linspace(-1, 1, n_sq, endpoint=False)
    vals_sq = sq_wave(t_sq)

    # Get Fourier coefficients
    coeffs_sq = np.array(trig_vals2coeffs(jnp.array(vals_sq, dtype=jnp.float64)))
    M_sq = n_sq // 2

    degree = 15
    trunc_c = np.zeros(2 * degree + 1, dtype=np.complex128)
    for k in range(-degree, degree + 1):
        trunc_c[k + degree] = coeffs_sq[k + M_sq]

    # Build truncated Trigtech
    u_trunc = Trigtech(coeffs=jnp.array(trunc_c), is_real=True, ishappy=True)

    xs_plot = np.linspace(-1, 1, 1000)
    ys_sq = sq_wave(xs_plot)
    ys_trunc = np.array(u_trunc(jnp.array(xs_plot)))

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(xs_plot * np.pi, ys_sq, color=CHEBFUN_BLUE, linewidth=1.8)
    ax.plot(xs_plot * np.pi, ys_trunc, color=CHEBFUN_RED, linewidth=1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "Gibbs phenomenon")
except Exception as e:
    plot_idx += 1
    print(f"  guide11_{plot_idx:02d}.png FAILED: {e}")

print(f"\nGuide 11: generated {plot_idx} plots.")
