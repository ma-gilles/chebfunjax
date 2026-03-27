"""Rounding corners by convolution.

Demonstrates how convolution with a smooth bump function rounds the
corners of piecewise linear curves. Translated from geom/RoundingCorners.m.

Original: https://www.chebfun.org/examples/geom/RoundingCorners.html
Author: Nick Trefethen, November 2012
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.signal import fftconvolve
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()



def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/geom')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(2, 2)

    n_pts = 5000
    t = np.linspace(-1, 1, n_pts)
    dt = t[1] - t[0]

    # W-shaped function: f = 3 * min(|t+0.4|, |t-0.3|)
    f = 3 * np.minimum(np.abs(t + 0.4), np.abs(t - 0.3))

    # Narrow kernel with integral 1: tent on [-h, h]
    h = 0.1
    s_full = np.linspace(-1, 1, n_pts)  # kernel on full domain
    g_kernel = np.where(np.abs(s_full) <= h, (h - np.abs(s_full)) / h**2, 0.0)
    # Normalize to integral 1
    g_kernel /= np.trapezoid(g_kernel, s_full)

    # Convolution of f with g
    f2_raw = fftconvolve(f, g_kernel, mode='same') * dt
    f2 = f2_raw  # already on same domain

    # Plot W function and kernel
    axes[0, 0].plot(t, f, 'b-', linewidth=2)
    axes[0, 0].set_title('W-shaped function f(t)', fontsize=11)
    axes[0, 0].set_xlabel('t'); axes[0, 0].set_xlim(-1.2, 1.2)
    axes[0, 0].set_ylim(0, 2.4); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_aspect('equal')

    axes[0, 1].plot(t, g_kernel, 'k-', linewidth=2)
    axes[0, 1].set_title(f'Smoothing kernel g(t) (h={h})', fontsize=11)
    axes[0, 1].set_xlabel('t'); axes[0, 1].set_xlim(-1, 1)
    axes[0, 1].grid(True, alpha=0.3)
    print(f"Kernel integral: {np.trapezoid(g_kernel, t):.6f}  (should be 1.0)")

    # Rounded W
    axes[1, 0].plot(t, f2, 'b-', linewidth=2)
    axes[1, 0].set_title('Rounded W: f * g', fontsize=11)
    axes[1, 0].set_xlabel('t'); axes[1, 0].set_xlim(-1.2, 1.2)
    axes[1, 0].set_ylim(0, 2.4); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect('equal')

    # Complex W curve: W(t) = t + i*f(t), convolved with g
    W = t + 1j * f
    W_real_conv = fftconvolve(np.real(W), g_kernel, mode='same') * dt
    W_imag_conv = fftconvolve(np.imag(W), g_kernel, mode='same') * dt
    W2 = W_real_conv + 1j * W_imag_conv

    axes[1, 1].plot(np.real(W), np.imag(W), 'b-', linewidth=2, label='Original W curve')
    axes[1, 1].plot(np.real(W2), np.imag(W2), 'r-', linewidth=2, label='Rounded W curve')
    axes[1, 1].set_title('Complex W curve before/after rounding', fontsize=10)
    axes[1, 1].set_xlabel('Re(W)'); axes[1, 1].set_ylabel('Im(W)')
    axes[1, 1].legend(fontsize=9); axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(-1.2, 1.2); axes[1, 1].set_ylim(0, 2.4)
    axes[1, 1].set_aspect('equal')

    fig.suptitle('Rounding Corners by Convolution', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'rounding_corners.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("rounding_corners: done")
    return True


if __name__ == "__main__":
    run()
