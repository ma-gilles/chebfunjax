"""Procrustes shape analysis.

Demonstrates shape comparison between curves by translating, scaling,
and rotating to canonical form. Translated from geom/Procrustes.m.

Original: https://www.chebfun.org/examples/geom/Procrustes.html
Author: Alex Townsend, August 2011
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj


def shape_analysis(f, g, n_pts=500):
    """Perform Procrustes shape analysis on two parametric curves.

    f, g: complex arrays of curve samples.
    Returns normalized, aligned curves and Procrustes distance.
    """
    # 1. Translate to mean zero
    f = f - np.mean(f)
    g = g - np.mean(g)

    # 2. Scale to unit RMSD
    f_norm = np.sqrt(np.mean(np.abs(f)**2))
    g_norm = np.sqrt(np.mean(np.abs(g)**2))
    f_scaled = f / f_norm
    g_scaled = g / g_norm

    # 3. Rotate to align: find rotation that minimizes |f - r*g|^2
    # Optimal rotation: r = exp(i*phi) where phi = -arg(sum f*conj(g))
    inner = np.sum(f_scaled * np.conj(g_scaled))
    phi = -np.angle(inner)
    g_aligned = g_scaled * np.exp(1j * phi)

    # Procrustes distance
    dist = np.sqrt(np.mean(np.abs(f_scaled - g_aligned)**2))
    return f, g, f_scaled, g_scaled, g_aligned, dist


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/geom')
    os.makedirs(outdir, exist_ok=True)

    t = np.linspace(0, 2 * np.pi, 500)

    # Frisbee: 3*(1.5*cos(t) + i*sin(t))
    f = 3 * (1.5 * np.cos(t) + 1j * np.sin(t))
    # Pebble: exp(i*pi/3)*(1+cos(t)+1.5i*sin(t)+.125*(1+1.5i)*sin(3t)^2)
    g = np.exp(1j * np.pi / 3) * (1 + np.cos(t) + 1.5j * np.sin(t)
                                    + 0.125 * (1 + 1.5j) * np.sin(3 * t)**2)

    f_orig, g_orig, f_scaled, g_scaled, g_aligned, dist = shape_analysis(f, g)

    print(f"Procrustes distance (frisbee vs pebble): {dist:.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Original
    axes[0, 0].plot(np.real(f), np.imag(f), 'r-', linewidth=2, label='Frisbee')
    axes[0, 0].plot(np.real(g), np.imag(g), 'k-', linewidth=2, label='Pebble')
    axes[0, 0].set_title('Original', fontsize=12)
    axes[0, 0].set_aspect('equal'); axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # After translation (mean = 0)
    axes[0, 1].plot(np.real(f_orig), np.imag(f_orig), 'r-', linewidth=2)
    axes[0, 1].plot(np.real(g_orig), np.imag(g_orig), 'k-', linewidth=2)
    axes[0, 1].set_title('After translation (mean=0)', fontsize=12)
    axes[0, 1].set_aspect('equal'); axes[0, 1].grid(True, alpha=0.3)

    # After scaling (RMSD=1)
    axes[1, 0].plot(np.real(f_scaled), np.imag(f_scaled), 'r-', linewidth=2)
    axes[1, 0].plot(np.real(g_scaled), np.imag(g_scaled), 'k-', linewidth=2)
    axes[1, 0].set_title('After scaling (RMSD=1)', fontsize=12)
    axes[1, 0].set_aspect('equal'); axes[1, 0].grid(True, alpha=0.3)

    # After rotation alignment
    axes[1, 1].plot(np.real(f_scaled), np.imag(f_scaled), 'r-', linewidth=2, label='Frisbee')
    axes[1, 1].plot(np.real(g_aligned), np.imag(g_aligned), 'k-', linewidth=2, label='Pebble')
    axes[1, 1].set_title(f'After alignment\nProcrustes dist = {dist:.4f}', fontsize=12)
    axes[1, 1].set_aspect('equal'); axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    # Compare to a more similar shape
    g2 = 1.1 * np.exp(1j * 0.3) * (1.5 * np.cos(t) + 1j * np.sin(t)) + 0.2
    _, _, f_s2, g_s2, g_a2, dist2 = shape_analysis(f, g2)
    print(f"Procrustes distance (frisbee vs near-frisbee): {dist2:.6f}")

    fig.suptitle('Procrustes Shape Analysis', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'procrustes.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("procrustes: done")
    return True


if __name__ == "__main__":
    run()
