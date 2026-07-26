"""Generate all plots for Guide Chapter 11: Periodic Chebfuns."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import matplotlib

matplotlib.use('Agg')

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from chebfunjax.plotting import CHEBFUN_BLUE, CHEBFUN_RED, chebfun_style
from chebfunjax.tech.trigtech import Trigtech, trig_vals2coeffs

chebfun_style()

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT_DIR, exist_ok=True)

plot_idx = 0


def save(fig, hint=""):
    global plot_idx
    plot_idx += 1
    path = os.path.join(OUT_DIR, f'guide11_{plot_idx:02d}.png')
    from chebfunjax.plotting import save_chebfun_figure
    save_chebfun_figure(fig, path, size=(600, 270))
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
    fig, ax = plt.subplots()
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

    fig, ax = plt.subplots()
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

    fig, ax = plt.subplots()
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
    fig, ax = plt.subplots()
    ax.plot(xs_phys, ys_abs, color=CHEBFUN_BLUE, linewidth=1.8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "|f| (abs of trigfun)")
except Exception as e:
    plot_idx += 1
    print(f"  guide11_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 5: starburst gallery example -- cheb.gallerytrig('starburst') (Sec 11.4)
#   fa(t) = (3 + sin(10t) + sin(61 exp(.8 sin t + .7))) exp(i t), a complex
#   trig chebfun; plot(f) draws imag(f) against real(f) with axis equal.
# --------------------------------------------------------------------------
try:
    starburst = lambda s: (
        (3.0 + jnp.sin(10.0 * jnp.pi * s)
         + jnp.sin(61.0 * jnp.exp(0.8 * jnp.sin(jnp.pi * s) + 0.7)))
        * jnp.exp(1j * jnp.pi * s)
    )
    f_star = Trigtech.from_function(starburst)
    nlen = len(np.array(f_star.coeffs))
    ss = np.linspace(-1, 1, 4000)
    zz = np.array(f_star(jnp.array(ss)))

    fig, ax = plt.subplots()
    ax.plot(zz.real, zz.imag, color=CHEBFUN_BLUE, linewidth=1.0)
    ax.set_aspect('equal')
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.0, 5.5)
    ax.set_title(f'starburst, length = {nlen}')
    save(fig, "starburst")
except Exception as e:
    plot_idx += 1
    print(f"  guide11_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 6: noisy periodic function (201 samples) as a trig chebfun  (Sec 11.5)
# --------------------------------------------------------------------------
try:
    np.random.seed(0)
    n_pts = 201
    tt = np.linspace(-np.pi, np.pi, n_pts, endpoint=False)
    ff_vals = np.exp(np.sin(tt)) + 0.05 * np.random.randn(n_pts)

    # Genuine trigonometric interpolant through the 201 equispaced samples.
    c_noisy = trig_vals2coeffs(jnp.array(ff_vals, dtype=jnp.complex128))
    f_noisy = Trigtech(coeffs=c_noisy, is_real=True, ishappy=True)
    xs = np.linspace(-1, 1, 2000)
    y_noisy = np.array(f_noisy(jnp.array(xs))).real

    fig, ax = plt.subplots()
    ax.plot(xs * np.pi, y_noisy, color=CHEBFUN_BLUE, linewidth=1.0)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "noisy function")
except Exception as e:
    plot_idx += 1
    print(f"  guide11_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 7: smoothed curve from circular convolution with a Gaussian  (Sec 11.5)
#   overlaid on the noisy function of plot 6 (hold on, plot(h)).
# --------------------------------------------------------------------------
try:
    sigma = 0.1
    gaussian = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (tt / sigma)**2)
    dt = 2 * np.pi / n_pts
    h_vals = np.real(np.fft.ifft(np.fft.fft(ff_vals) * np.fft.fft(gaussian))) * dt
    # roll so the Gaussian (peaked at t=0) convolves centred on the grid
    h_vals = np.roll(h_vals, n_pts // 2)
    c_smooth = trig_vals2coeffs(jnp.array(h_vals, dtype=jnp.complex128))
    f_smooth = Trigtech(coeffs=c_smooth, is_real=True, ishappy=True)
    y_smooth = np.array(f_smooth(jnp.array(xs))).real

    fig, ax = plt.subplots()
    ax.plot(xs * np.pi, y_noisy, color=CHEBFUN_BLUE, linewidth=1.0)
    ax.plot(xs * np.pi, y_smooth, color=CHEBFUN_RED, linewidth=1.8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "noisy + Gaussian-smoothed")
except Exception as e:
    plot_idx += 1
    print(f"  guide11_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 8: plotcoeffs for exp(sin(t)) -- supergeometric decay  (Section 11.7)
# --------------------------------------------------------------------------
try:
    f_ent = Trigtech.from_function(lambda s: jnp.exp(jnp.sin(jnp.pi * s)))
    c = np.abs(np.array(f_ent.coeffs))
    c = np.maximum(c, 1e-18)
    ks = np.arange(len(c)) - len(c) // 2

    fig, ax = plt.subplots()
    ax.semilogy(ks, c, '.', color=CHEBFUN_BLUE, markersize=4)
    ax.set_title('Fourier coefficients')
    ax.set_xlabel('Wave number')
    ax.set_ylabel('Magnitude of coefficient')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "plotcoeffs exp(sin(t))")
except Exception as e:
    plot_idx += 1
    print(f"  guide11_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 9: plotcoeffs for 1/(2-cos(t)) -- geometric decay  (Section 11.7)
# --------------------------------------------------------------------------
try:
    f_geo = Trigtech.from_function(lambda s: 1.0 / (2.0 - jnp.cos(jnp.pi * s)))
    c = np.abs(np.array(f_geo.coeffs))
    c = np.maximum(c, 1e-18)
    ks = np.arange(len(c)) - len(c) // 2

    fig, ax = plt.subplots()
    ax.semilogy(ks, c, '.', color=CHEBFUN_BLUE, markersize=4)
    ax.set_title('Fourier coefficients')
    ax.set_xlabel('Wave number')
    ax.set_ylabel('Magnitude of coefficient')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "plotcoeffs 1/(2-cos(t))")
except Exception as e:
    plot_idx += 1
    print(f"  guide11_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 10: plotcoeffs for |sin(t)|^5 -- algebraic decay  (Section 11.7)
# --------------------------------------------------------------------------
try:
    f_alg = Trigtech.from_function(lambda s: jnp.abs(jnp.sin(jnp.pi * s))**5)
    c = np.abs(np.array(f_alg.coeffs))
    c = np.maximum(c, 1e-18)
    ks = np.arange(len(c)) - len(c) // 2

    fig, ax = plt.subplots()
    ax.semilogy(ks, c, '.', color=CHEBFUN_BLUE, markersize=4)
    ax.set_title('Fourier coefficients')
    ax.set_xlabel('Wave number')
    ax.set_ylabel('Magnitude of coefficient')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "plotcoeffs |sin(t)|^5")
except Exception as e:
    plot_idx += 1
    print(f"  guide11_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 11: plotcoeffs(f,'loglog') for |sin(t)|^5 with k^{-6} reference (Sec 11.7)
# --------------------------------------------------------------------------
try:
    # |sin(t)|^5 on [-pi,pi] in trig mode via the factory
    import chebfunjax as cj
    PI = float(np.pi)
    f5 = cj.chebfun(lambda t: jnp.abs(jnp.sin(t))**5,
                    domain=[-PI, PI], trig=True)
    c5 = np.abs(np.asarray(f5.funs[0].tech.coeffs))
    n5 = len(c5)
    half5 = (n5 - 1) // 2
    ks = np.abs(np.arange(-half5, half5 + 1)) + 1.0  # |wave number| + 1

    fig, ax = plt.subplots()
    ax.loglog(ks, np.maximum(c5, 1e-300), '.', color=CHEBFUN_BLUE,
              markersize=3)
    ks_ref = np.array([9.0, 900.0])
    ax.loglog(ks_ref, 3.0 * ks_ref**(-6), '--r', linewidth=1.4)
    ax.text(110, 4e-9, '$k^{-6}$', color='r', fontsize=11)
    ax.set_title('Fourier coefficients')
    ax.set_xlabel('|Normalized wave number|+1')
    ax.set_ylabel('Magnitude of coefficient')
    ax.set_xlim(1, 1e3)
    ax.grid(True, which='both', alpha=0.3, linestyle=':', linewidth=0.5)
    save(fig, "loglog coeffs with k^-6 line")
except Exception as e:
    plot_idx += 1
    print(f"  guide11_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 12: square wave and degree-15 truncated Fourier series  (Sec 11.8)
# --------------------------------------------------------------------------
try:
    import chebfunjax as cj
    PI = float(np.pi)
    # sign(sin t): piecewise via explicit breakpoints (no edge detection
    # needed for this known jump location)
    u = cj.chebfun(lambda t: jnp.where(t < 0.0, -1.0, 1.0),
                   domain=[-PI, 0.0, PI])

    # trigcoeffs(u, 31) equivalent: c_k = (1/2pi) * int u(t) e^{-ikt} dt,
    # computed honestly by quadrature through the piecewise chebfun.
    degree = 15
    ks12 = np.arange(-degree, degree + 1)
    coeffs12 = []
    for k in ks12:
        ck = complex((u * cj.chebfun(
            lambda t, _k=float(k): jnp.exp(-1j * _k * t),
            domain=[-PI, 0.0, PI])).sum()) / (2 * PI)
        coeffs12.append(ck)
    coeffs12 = np.array(coeffs12)

    ts = np.linspace(-PI, PI, 1200)
    u_trunc = np.real(np.exp(1j * np.outer(ts, ks12)) @ coeffs12)

    fig, ax = plt.subplots()
    # square wave in blue with dotted jump (draw per piece)
    ax.plot([-PI, 0], [-1, -1], color=CHEBFUN_BLUE, linewidth=1.6)
    ax.plot([0, PI], [1, 1], color=CHEBFUN_BLUE, linewidth=1.6)
    ax.plot([0, 0], [-1, 1], ':', color=CHEBFUN_BLUE, linewidth=1.2)
    ax.plot(ts, u_trunc, color=CHEBFUN_RED, linewidth=1.4)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(-PI, PI)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "square wave + truncated Fourier")
except Exception as e:
    plot_idx += 1
    print(f"  guide11_{plot_idx:02d}.png FAILED: {e}")

print(f"\nGuide 11: generated {plot_idx} plots.")
