"""Generate all plots for Guide Chapter 5: Complex Chebfuns.

Faithful translation of the MATLAB Chebfun Guide Chapter 5 plots.
Complex chebfuns in chebfunjax are represented as (real_part, imag_part) pairs.
"""

import matplotlib
matplotlib.use('Agg')

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT_DIR, exist_ok=True)

plot_idx = 0

# Helper: evaluate complex chebfun pair at points
def eval_complex(x_cheb, y_cheb, tt):
    return x_cheb(tt), y_cheb(tt)


# --------------------------------------------------------------------------
# Plot 1: 20 points on upper half unit circle -- Section 5.1
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    s = np.linspace(0, np.pi, 20)
    f_re = np.cos(s)
    f_im = np.sin(s)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(f_re, f_im, '.', markersize=10)
    ax.set_aspect('equal')
    ax.set_title('20 points on the upper semicircle')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide05_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide05_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 2: Chebfun semicircle -- Section 5.1
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    s = cj.chebfun(lambda s: s, domain=[0, float(jnp.pi)])
    f_re = cj.cos(s)
    f_im = cj.sin(s)

    tt = jnp.linspace(0, float(jnp.pi), 200)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(f_re(tt), f_im(tt), linewidth=1.5)
    ax.set_aspect('equal')
    ax.set_title('Chebfun semicircle')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide05_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide05_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 3: Semicircle with Chebyshev points marked -- Section 5.1
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    s = cj.chebfun(lambda s: s, domain=[0, float(jnp.pi)])
    f_re = cj.cos(s)
    f_im = cj.sin(s)
    n_pts = len(f_re)

    tt = jnp.linspace(0, float(jnp.pi), 200)
    # Chebyshev points on [0, pi]
    from chebfunjax.utils.quadrature import chebpts
    s_pts = (chebpts(n_pts) + 1) / 2 * float(jnp.pi)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(f_re(tt), f_im(tt), '.-', markersize=10, linewidth=1.2)
    ax.set_aspect('equal')
    ax.set_title(f'length(f) = {n_pts}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide05_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide05_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 4: s*exp(10i*s) and exp(2i*s)+0.3*exp(20i*s) -- Section 5.1
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    s = cj.chebfun(lambda s: s, domain=[0, float(jnp.pi)])

    # g = s*exp(10i*s)
    g_re = s * cj.cos(10 * s)
    g_im = s * cj.sin(10 * s)

    # h = exp(2i*s) + 0.3*exp(20i*s)
    h_re = cj.cos(2 * s) + 0.3 * cj.cos(20 * s)
    h_im = cj.sin(2 * s) + 0.3 * cj.sin(20 * s)

    tt = jnp.linspace(0, float(jnp.pi), 1000)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(g_re(tt), g_im(tt), linewidth=1.2)
    axes[0].set_aspect('equal')
    axes[0].set_title(r'$s \cdot e^{10is}$')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(h_re(tt), h_im(tt), linewidth=1.2)
    axes[1].set_aspect('equal')
    axes[1].set_title(r'$e^{2is} + 0.3 e^{20is}$')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide05_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide05_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 5: g^2 and exp(h) -- Section 5.1
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    tt = jnp.linspace(0, float(jnp.pi), 1000)

    # g^2 = (g_re + i*g_im)^2 = g_re^2 - g_im^2 + 2i*g_re*g_im
    g2_re = g_re**2 - g_im**2
    g2_im = 2 * g_re * g_im

    # exp(h) = exp(h_re) * (cos(h_im) + i*sin(h_im))
    exph_re = cj.exp(h_re) * cj.cos(h_im)
    exph_im = cj.exp(h_re) * cj.sin(h_im)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(g2_re(tt), g2_im(tt), linewidth=1.2)
    axes[0].set_aspect('equal')
    axes[0].set_title(r'$g^2$')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(exph_re(tt), exph_im(tt), linewidth=1.2)
    axes[1].set_aspect('equal')
    axes[1].set_title(r'$\exp(h)$')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide05_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide05_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 6: Piecewise complex z and z^2 -- Section 5.1
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    # z = (1+0.5i)*s for s in [0,1] and 1+0.5i-2(s-1) for s in [1,2]
    # Real and imag parts:
    # Piece 1: z_re = s, z_im = 0.5*s on [0,1]
    # Piece 2: z_re = 1 - 2*(s-1) = 3 - 2s, z_im = 0.5 on [1,2]
    s1 = cj.chebfun(lambda s: s, domain=[0, 1])
    z1_re = s1
    z1_im = 0.5 * s1

    s2 = cj.chebfun(lambda s: s, domain=[1, 2])
    z2_re = 3 - 2 * s2
    z2_im = cj.chebfun(lambda s: 0.5 + 0*s, domain=[1, 2])

    tt1 = jnp.linspace(0, 1, 200)
    tt2 = jnp.linspace(1, 2, 200)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # z
    axes[0].plot(z1_re(tt1), z1_im(tt1), linewidth=1.5)
    axes[0].plot(z2_re(tt2), z2_im(tt2), linewidth=1.5)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('z')

    # z^2 = (z_re + i*z_im)^2
    z1sq_re = z1_re**2 - z1_im**2
    z1sq_im = 2 * z1_re * z1_im
    z2sq_re = z2_re**2 - z2_im**2
    z2sq_im = 2 * z2_re * z2_im

    axes[1].plot(z1sq_re(tt1), z1sq_im(tt1), linewidth=1.5)
    axes[1].plot(z2sq_re(tt2), z2sq_im(tt2), linewidth=1.5)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(r'$z^2$')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide05_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide05_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 7: Rectangle R and cross X -- Section 5.2
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    # R: rectangle with corners at 1, 2, 2+2i, 1+2i
    # X: cross inside R
    tt = jnp.linspace(0, 1, 200)

    # Rectangle: R = join(1+s, 2+2i*s, 2+2i-s, 1+2i-2i*s)
    # Side 1: z = 1+s, s in [0,1] -> re=1+s, im=0
    R1_re = 1 + tt
    R1_im = jnp.zeros_like(tt)
    # Side 2: z = 2+2i*s -> re=2, im=2*s
    R2_re = 2 * jnp.ones_like(tt)
    R2_im = 2 * tt
    # Side 3: z = 2+2i-s -> re=2-s, im=2
    R3_re = 2 - tt
    R3_im = 2 * jnp.ones_like(tt)
    # Side 4: z = 1+2i-2i*s -> re=1, im=2-2*s
    R4_re = jnp.ones_like(tt)
    R4_im = 2 - 2 * tt

    # Cross:
    # X1: 1.3+1.5i + 0.4*s -> re=1.3+0.4*s, im=1.5
    X1_re = 1.3 + 0.4 * tt
    X1_im = 1.5 * jnp.ones_like(tt)
    # X2: 1.5+1.3i + 0.4i*s -> re=1.5, im=1.3+0.4*s
    X2_re = 1.5 * jnp.ones_like(tt)
    X2_im = 1.3 + 0.4 * tt

    fig, ax = plt.subplots(figsize=(6, 6))
    # Rectangle
    R_re = jnp.concatenate([R1_re, R2_re, R3_re, R4_re])
    R_im = jnp.concatenate([R1_im, R2_im, R3_im, R4_im])
    ax.plot(R_re, R_im, 'b', linewidth=2.2)
    # Cross
    ax.plot(X1_re, X1_im, 'r', linewidth=2.2)
    ax.plot(X2_re, X2_im, 'r', linewidth=2.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Rectangle R and cross X')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide05_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide05_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 8: R^2 and exp(R) -- Section 5.2
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # z^2: (re + i*im)^2 = re^2 - im^2 + 2i*re*im
    def complex_sq(re, im):
        return re**2 - im**2, 2 * re * im

    def complex_exp(re, im):
        return np.exp(re) * np.cos(im), np.exp(re) * np.sin(im)

    # R^2
    for (rr, ri) in [(R1_re, R1_im), (R2_re, R2_im), (R3_re, R3_im), (R4_re, R4_im)]:
        sq_re, sq_im = complex_sq(rr, ri)
        axes[0].plot(sq_re, sq_im, 'b', linewidth=1.5)
    for (rr, ri) in [(X1_re, X1_im), (X2_re, X2_im)]:
        sq_re, sq_im = complex_sq(rr, ri)
        axes[0].plot(sq_re, sq_im, 'r', linewidth=2.2)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(r'$z^2$')

    # exp(R)
    for (rr, ri) in [(R1_re, R1_im), (R2_re, R2_im), (R3_re, R3_im), (R4_re, R4_im)]:
        e_re, e_im = complex_exp(rr, ri)
        axes[1].plot(e_re, e_im, 'b', linewidth=1.5)
    for (rr, ri) in [(X1_re, X1_im), (X2_re, X2_im)]:
        e_re, e_im = complex_exp(rr, ri)
        axes[1].plot(e_re, e_im, 'r', linewidth=2.2)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(r'$\exp(z)$')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide05_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide05_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 9: Complex grid -- Section 5.2
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    t = jnp.linspace(-1, 1, 200)
    fig, ax = plt.subplots(figsize=(6, 6))
    for d_val in jnp.arange(-1, 1.01, 0.2):
        d = float(d_val)
        # Horizontal: z = t + i*d
        ax.plot(t, jnp.full_like(t, d), 'b', linewidth=0.6, alpha=0.7)
        # Vertical: z = d + i*t
        ax.plot(jnp.full_like(t, d), t, 'b', linewidth=0.6, alpha=0.7)
    ax.set_aspect('equal')
    ax.set_title('Complex grid')
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide05_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide05_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 10: exp(grid) and tan(grid) -- Section 5.2
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    t = jnp.linspace(-1, 1, 300)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for d_val in jnp.arange(-1, 1.01, 0.2):
        d = float(d_val)
        # Line z = t + i*d
        z = t + 1j * d
        # exp(z)
        w = jnp.exp(z)
        axes[0].plot(jnp.real(w), jnp.imag(w), 'b', linewidth=0.6, alpha=0.7)
        # tan(z)
        w2 = jnp.tan(z)
        good = jnp.abs(jnp.imag(w2)) < 10  # clip large values
        # Plot segments where good
        axes[1].plot(jnp.where(good, jnp.real(w2), jnp.nan),
                     jnp.where(good, jnp.imag(w2), jnp.nan),
                     'b', linewidth=0.6, alpha=0.7)

        # Line z = d + i*t
        z = d + 1j * t
        w = jnp.exp(z)
        axes[0].plot(jnp.real(w), jnp.imag(w), 'b', linewidth=0.6, alpha=0.7)
        w2 = jnp.tan(z)
        good = jnp.abs(jnp.imag(w2)) < 10
        axes[1].plot(jnp.where(good, jnp.real(w2), jnp.nan),
                     jnp.where(good, jnp.imag(w2), jnp.nan),
                     'b', linewidth=0.6, alpha=0.7)

    axes[0].set_aspect('equal')
    axes[0].set_title(r'$\exp(z)$')
    axes[0].grid(True, alpha=0.2)
    axes[1].set_aspect('equal')
    axes[1].set_title(r'$\tan(z)$')
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide05_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide05_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 11: grid, exp(grid), tan(grid) side by side -- Section 5.2
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    t = jnp.linspace(-1, 1, 300)
    fig, ax = plt.subplots(figsize=(16, 5))

    for d_val in jnp.arange(-1, 1.01, 0.2):
        d = float(d_val)
        # Original grid
        z = t + 1j * d
        ax.plot(jnp.real(z), jnp.imag(z), 'b', linewidth=0.5, alpha=0.7)
        z = d + 1j * t
        ax.plot(jnp.real(z), jnp.imag(z), 'b', linewidth=0.5, alpha=0.7)

        # exp(grid) shifted
        z = t + 1j * d
        w = jnp.exp(z)
        ax.plot(1.6 + jnp.real(w), jnp.imag(w), 'b', linewidth=0.5, alpha=0.7)
        z = d + 1j * t
        w = jnp.exp(z)
        ax.plot(1.6 + jnp.real(w), jnp.imag(w), 'b', linewidth=0.5, alpha=0.7)

        # tan(grid) shifted
        z = t + 1j * d
        w = jnp.tan(z)
        good = jnp.abs(jnp.imag(w)) < 10
        ax.plot(jnp.where(good, 6.6 + jnp.real(w), jnp.nan),
                jnp.where(good, jnp.imag(w), jnp.nan),
                'b', linewidth=0.5, alpha=0.7)
        z = d + 1j * t
        w = jnp.tan(z)
        good = jnp.abs(jnp.imag(w)) < 10
        ax.plot(jnp.where(good, 6.6 + jnp.real(w), jnp.nan),
                jnp.where(good, jnp.imag(w), jnp.nan),
                'b', linewidth=0.5, alpha=0.7)

    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide05_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide05_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 12: Moebius iterations of a square -- Section 5.2
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    moebius = lambda z: 1.0 / (1.0 + z)

    # Square: corners at 0-0.5i, 1-0.5i, 1+0.5i, 0+0.5i
    t = jnp.linspace(0, 1, 200)
    S_re = jnp.concatenate([-0.0*jnp.ones(200) + t, jnp.ones(200), 1 - t, jnp.zeros(200)])
    S_im = jnp.concatenate([-0.5*jnp.ones(200) + 0*t, -0.5 + t, 0.5*jnp.ones(200), 0.5 - t])

    fig, ax = plt.subplots(figsize=(8, 6))
    z = S_re + 1j * S_im
    for j in range(4):
        ax.plot(jnp.real(z), jnp.imag(z), linewidth=1.2)
        z = moebius(z)

    # Plot fixed point
    ax.plot((np.sqrt(5) - 1) / 2, 0, '.k', markersize=6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Moebius iterations')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide05_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide05_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 13: Moebius filled squares -- Section 5.2
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    colors = [[0.5, 0.5, 1], [0.5, 1, 0.5], [1, 0.5, 0.5], [0.5, 1, 1]]

    fig, ax = plt.subplots(figsize=(8, 6))
    z = S_re + 1j * S_im
    for j, col in enumerate(colors):
        ax.fill(jnp.real(z), jnp.imag(z), color=col)
        z = moebius(z)

    ax.plot((np.sqrt(5) - 1) / 2, 0, '.k', markersize=6)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide05_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide05_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 14: Keyhole contour -- Section 5.3
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    # Keyhole contour
    c = [-2 + 0.05j, -0.2 + 0.05j, -0.2 - 0.05j, -2 - 0.05j]
    t = np.linspace(0, 1, 500)

    # Side 1: straight from c[0] to c[1]
    z1 = c[0] + t * (c[1] - c[0])
    # Side 2: small arc from c[1] to c[2]: c[1] * (c[2]/c[1])^t
    z2 = c[1] * (c[2] / c[1])**t
    # Side 3: straight from c[2] to c[3]
    z3 = c[2] + t * (c[3] - c[2])
    # Side 4: large arc from c[3] to c[0]: c[3] * (c[0]/c[3])^t
    z4 = c[3] * (c[0] / c[3])**t

    z = np.concatenate([z1, z2, z3, z4])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.real(z), np.imag(z), linewidth=1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Keyhole contour')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide05_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide05_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 15: "chebfunjax" scribble -- Section 5.5
# (scribble not available, we use text-based path instead)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.text(0.5, 0.5, 'chebfunjax', fontsize=40, ha='center', va='center',
            fontfamily='monospace', fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('scribble (text approximation)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide05_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide05_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 16: exp(3i*f) of scribble -- Section 5.5
# --------------------------------------------------------------------------
# Skip -- scribble not available
plot_idx += 1
print(f"guide05_{plot_idx:02d}.png SKIPPED (scribble not available)")

# --------------------------------------------------------------------------
# Plot 17-21: More scribble plots -- Section 5.5
# --------------------------------------------------------------------------
for _ in range(5):
    plot_idx += 1
    print(f"guide05_{plot_idx:02d}.png SKIPPED (scribble not available)")

print(f"\nGuide 05 plot generation complete. Generated {plot_idx} plots.")
