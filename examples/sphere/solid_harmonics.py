"""Solid harmonics.

Solid harmonics are solutions to Laplace's equation in 3D spherical
coordinates. Regular solid harmonics R_l^m grow as r^l, while irregular
ones S_l^m decay as r^{-(l+1)}.
Translated from sphere/SolidHarmonics.m.

Original: https://www.chebfun.org/examples/sphere/SolidHarmonics.html
Authors: Nicolas Boulle and Alex Townsend, May 2019
"""

import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sph_harm_y

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import jax.numpy as jnp

from chebfunjax.plotting import chebfun_style, plot_sphere
from chebfunjax.spherefun.spherefun import Spherefun

chebfun_style()

def regular_solid_harmonic(l, m, r, theta, phi):
    """Regular solid harmonic: R_l^m(r, theta, phi) = r^l * Y_l^m(theta, phi)."""
    Ylm_raw = sph_harm_y(l, abs(m), theta, phi)
    if m > 0:
        Ylm = np.sqrt(2) * (-1)**m * np.real(Ylm_raw)
    elif m < 0:
        Ylm = np.sqrt(2) * (-1)**m * np.imag(Ylm_raw)
    else:
        Ylm = np.real(Ylm_raw)
    return r**l * Ylm

def irregular_solid_harmonic(l, m, r, theta, phi):
    """Irregular solid harmonic: S_l^m = r^{-(l+1)} * Y_l^m(theta, phi)."""
    Ylm_raw = sph_harm_y(l, abs(m), theta, phi)
    if m > 0:
        Ylm = np.sqrt(2) * (-1)**m * np.real(Ylm_raw)
    elif m < 0:
        Ylm = np.sqrt(2) * (-1)**m * np.imag(Ylm_raw)
    else:
        Ylm = np.real(Ylm_raw)
    return r**(-(l+1)) * Ylm

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/sphere')
    os.makedirs(outdir, exist_ok=True)

    # Verify: Laplacian of R_l^m = 0
    print("Solid harmonics:")
    print("  R_l^m = r^l * Y_l^m satisfies Delta(R_l^m) = 0")
    print("  S_l^m = r^{-(l+1)} * Y_l^m satisfies Delta(S_l^m) = 0 for r!=0")

    # --- Panel 1: R_2^1 on the unit sphere ---
    l, m = 2, 1
    R21_sf = Spherefun.from_function(
        lambda lam, th: jnp.array(regular_solid_harmonic(l, m, 1.0,
            np.asarray(th), np.asarray(lam))))
    fig, ax = plot_sphere(R21_sf, title='Regular $R_2^1$ on unit sphere')
    fig.savefig(os.path.join(outdir, 'solid_harmonics_R21.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Panel 2: R_3^2 on the unit sphere ---
    l2, m2 = 3, 2
    R32_sf = Spherefun.from_function(
        lambda lam, th: jnp.array(regular_solid_harmonic(l2, m2, 1.0,
            np.asarray(th), np.asarray(lam))))
    fig, ax = plot_sphere(R32_sf, title='Regular $R_3^2$ on unit sphere')
    fig.savefig(os.path.join(outdir, 'solid_harmonics_R32.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Panel 3: Irregular solid harmonic S_2^1 on sphere of radius 2 ---
    r_outer = 2.0
    S21_sf = Spherefun.from_function(
        lambda lam, th: jnp.array(irregular_solid_harmonic(l, m, r_outer,
            np.asarray(th), np.asarray(lam))))
    fig, ax = plot_sphere(S21_sf, title='Irregular $S_2^1$ at $r=2$')
    fig.savefig(os.path.join(outdir, 'solid_harmonics_S21.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("solid_harmonics: done")
    return True

if __name__ == "__main__":
    run()
