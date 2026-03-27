"""A Taylor's theorem analogue for Chebyshev series.

Demonstrates convergence of Chebyshev approximation in Bernstein ellipses:
for entire functions, convergence is global; for functions with singularities,
convergence is limited to the largest Bernstein ellipse inside which the
function is analytic.
Translated from temp/TaylorsTheorem.m (original: approx/TaylorsTheorem.m).

Original: https://www.chebfun.org/examples/approx/TaylorsTheorem.html
Authors: Hrothgar and Anthony Austin, February 2015
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()



def cheb_interp(f, n, a=-1.0, b=1.0):
    """Chebyshev interpolant of f on [a,b] at n+1 points."""
    k = np.arange(n + 1)
    x_cheb = np.cos(np.pi * k / n)  # in [-1, 1]
    x_orig = (b - a) / 2 * x_cheb + (a + b) / 2
    f_vals = f(x_orig)
    # Chebyshev coefficients via DCT
    c = np.fft.rfft(np.concatenate([f_vals, f_vals[-2:0:-1]])).real / n
    c[1:-1] *= 2
    # Barycentric interpolation weights
    w = (-1)**k
    w[0] /= 2; w[-1] /= 2

    def interp(x_eval):
        result = np.zeros_like(x_eval, dtype=float)
        for i, xe in enumerate(x_eval):
            xe_norm = 2 * (xe - a) / (b - a) - 1
            diffs = xe_norm - x_cheb
            mask = np.abs(diffs) < 1e-14
            if np.any(mask):
                result[i] = f_vals[np.where(mask)[0][0]]
            else:
                result[i] = np.sum(w * f_vals / diffs) / np.sum(w / diffs)
        return result

    return interp, f_vals


def bernstein_ellipse(rho, n=200):
    """Points on the Bernstein ellipse with parameter rho in the complex plane."""
    theta = np.linspace(0, 2 * np.pi, n)
    z = rho * np.exp(1j * theta)
    # Joukowski map
    w = (z + 1.0 / z) / 2.0
    return w


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/temp')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel 1: Entire function sin(x) — global convergence ---
    f_sin = np.sin
    x_full = np.linspace(-7, 7, 500)

    axes[0].plot(x_full, np.sin(x_full), 'k-', linewidth=1.5, label='sin(x)')

    # Chebyshev approximants on [-pi/2, pi/2] with increasing degree
    a0, b0 = -np.pi/2, np.pi/2
    for k in range(2, 9):
        n_k = 2*k + 1
        interp_k, _ = cheb_interp(f_sin, n_k, a0, b0)
        y_k = interp_k(x_full)
        shade = 0.5 - k/20
        axes[0].plot(x_full, y_k, '-', color=(shade, 1-k/10, shade),
                     linewidth=1.0, alpha=0.7)

    axes[0].set_xlim(-7, 7); axes[0].set_ylim(-3, 3)
    axes[0].set_aspect('equal')
    axes[0].set_title('sin(x): entire function\nGlobal convergence', fontsize=10)
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('x')
    print("Panel 1: sin(x) Chebyshev approximants converge globally")

    # --- Panel 2: Non-entire function log|x-i| ---
    sing = 1j  # singularity at z=i

    def func(x):
        return np.log(np.abs(x - 1j))

    x_full2 = np.linspace(-5, 5, 500)
    axes[1].plot(x_full2, func(x_full2), 'k-', linewidth=1.5, label='log|x-i|')

    x0 = 2.0
    # Two intervals with different radii
    for r, col in [(0.5, 'red'), (1.2, 'blue')]:
        a_r, b_r = x0 - r, x0 + r
        interp_r, _ = cheb_interp(func, 30, a_r, b_r)
        y_r = interp_r(x_full2)
        axes[1].plot(x_full2, y_r, '-', color=col, linewidth=1.5,
                     alpha=0.8, label=f'r={r}')
        # Convergence boundary
        inv_J = lambda z: z + np.sqrt(z**2 - 1)
        rho_val = abs(inv_J((sing - x0) / r))
        d_val = abs((rho_val + 1/rho_val)/2) * r
        axes[1].axvline(x0 - d_val, linestyle='--', color=col, linewidth=1, alpha=0.7)
        axes[1].axvline(x0 + d_val, linestyle='--', color=col, linewidth=1, alpha=0.7)

    axes[1].set_ylim(-0.5, 2.5); axes[1].set_xlim(-5, 5)
    axes[1].set_title('log|x-i|: singularity at i\nConvergence limited to Bernstein ellipse',
                       fontsize=9)
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('x')
    print(f"Panel 2: log|x-i| with singularity at z=i")

    # --- Panel 3: Bernstein ellipses in complex plane ---
    ax3 = axes[2]
    ax3.set_aspect('equal')

    # Draw the interval [-1, 1]
    ax3.plot([-1, 1], [0, 0], 'k-', linewidth=3, label='Interval [-1,1]')

    # Draw Bernstein ellipses for increasing rho
    colors3 = plt.cm.Blues(np.linspace(0.3, 0.9, 6))
    rhos = np.linspace(1.2, 3.5, 6)
    inv_J = lambda z: z + np.sqrt(z**2 - 1 + 0j)
    for rho, col in zip(rhos, colors3):
        ellipse = bernstein_ellipse(rho)
        ax3.plot(np.real(ellipse), np.imag(ellipse), '-', color=col,
                 linewidth=1.5, alpha=0.8)

    # Mark singularity
    x0_3 = 2.0; sing_3 = 1j
    ax3.plot(np.real(sing_3), np.imag(sing_3), 'k*', markersize=12,
             label='Singularity', zorder=5)

    ax3.set_xlim(-4, 4); ax3.set_ylim(-3, 3)
    ax3.set_title('Bernstein ellipses\n(light=small ρ, dark=large ρ)', fontsize=10)
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Re(z)'); ax3.set_ylabel('Im(z)')

    print("Panel 3: Bernstein ellipses for various ρ values")
    print(f"  Joukowski map: z -> (z + 1/z)/2")

    fig.suptitle("Taylor's Theorem Analogue for Chebyshev Series", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'taylors_theorem.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("taylors_theorem: done")
    return True


if __name__ == "__main__":
    run()
