"""Rouche's theorem.

Rouche's theorem states: if |f(z)| > |g(z)| on a closed contour C, then
f and f+g have the same number of zeros inside C.  We verify this by
counting zeros using the argument principle.

Credit: Inspired by Chebfun example complex/RoucheTheorem.m
(Anthony Austin, November 2012, revised October 2020).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj


def count_zeros_inside(f_complex, r, n_pts=4000):
    """Count zeros of f inside |z| = r using the argument principle.

    Number of zeros = (1/(2*pi*i)) * int_{|z|=r} f'(z)/f(z) dz
                    = (1/(2*pi)) * Delta(arg f) around the contour
    """
    ts = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    zs = r * np.exp(1j * ts)
    fs = f_complex(zs)
    # Argument of f on the contour
    args = np.unwrap(np.angle(fs))
    delta_arg = args[-1] - args[0]
    # Winding number
    n_zeros = round(delta_arg / (2 * np.pi))
    return n_zeros


def run():
    print("=" * 60)
    print("Rouche's theorem")
    print("=" * 60)

    pi = float(jnp.pi)

    # --- Example 1: z^5 + 3*z^2 + 1 inside |z| = 2 -------------------
    # f = z^5, g = 3*z^2 + 1
    # On |z|=2: |f| = 32, |g| <= 3*4+1 = 13 < 32
    # So f+g has same number of zeros as f^5 inside |z|=2, which is 5
    def p1(z):
        return z**5 + 3*z**2 + 1

    n_zeros1 = count_zeros_inside(p1, 2.0)
    print(f"\nExample 1: p(z) = z^5 + 3*z^2 + 1 inside |z|=2")
    print(f"  Number of zeros: {n_zeros1}  (expected 5 by Rouche)")
    assert n_zeros1 == 5, f"Expected 5 zeros, got {n_zeros1}"

    # Verify by finding roots directly with numpy
    roots1 = np.roots([1, 0, 0, 3, 0, 1])  # coefficients from z^5 to z^0
    inside1 = roots1[np.abs(roots1) < 2.0]
    print(f"  Direct root count (numpy): {len(inside1)}")
    assert len(inside1) == 5

    # --- Example 2: z^4 + z^3 - 1 inside |z| = 1 ---------------------
    # f = z^3, g = z^4 - 1
    # On |z|=1: |f| = 1, |g| <= 1+1 = 2. So we need a different comparison.
    # Actually use f = -1 (constant), g = z^4 + z^3
    # |f| = 1, |g| <= 1+1 = 2 on |z|=1. This doesn't work directly.
    # Let's count directly:
    def p2(z):
        return z**4 + z**3 - 1

    n_zeros2 = count_zeros_inside(p2, 1.0)
    roots2 = np.roots([1, 1, 0, 0, -1])
    inside2 = roots2[np.abs(roots2) < 1.0 - 1e-10]
    print(f"\nExample 2: p(z) = z^4 + z^3 - 1 inside |z|=1")
    print(f"  Argument principle count: {n_zeros2}")
    print(f"  Direct root count: {len(inside2)}")
    assert n_zeros2 == len(inside2), f"Counts disagree: {n_zeros2} vs {len(inside2)}"

    # --- Example 3: Verify Rouche directly ---------------------------
    # Classic example: z^7 - 5*z^4 + 12 inside |z|=2
    # f = 5*z^4, g = z^7 + 12
    # On |z|=2: |f|=5*16=80, |g| <= 128+12=140. Doesn't work. Try |z|=1.5.
    # On |z|=1.5: |f|=5*(1.5^4)=25.3, |g|<=1.5^7+12=29.2. Doesn't work.
    # Use p(z) = z^7 - 5*z^4 + 12 from MATLAB Chebfun example.
    def p3(z):
        return z**7 - 5*z**4 + 12

    # Count zeros in concentric circles
    for R in [1.0, 1.5, 2.0, 3.0]:
        n = count_zeros_inside(p3, R)
        print(f"  p3(z) = z^7 - 5z^4 + 12 inside |z|={R}: {n} zeros")

    roots3 = np.roots([1, 0, 0, -5, 0, 0, 0, 12])
    print(f"  All roots: {np.sort(np.abs(roots3))}")

    # --- Example 4: Verify Rouche with linear perturbation ------------
    # For p(z) = z^n + epsilon * q(z) on |z|=r >> 1:
    # |z^n| >> |epsilon*q(z)| so p has n zeros inside by Rouche
    n = 6
    eps = 0.1

    def p4(z):
        return z**n + eps * (z**3 + 2*z + 3)

    n4 = count_zeros_inside(p4, 2.0)
    print(f"\nExample 4: z^{n} + 0.1*(z^3 + 2z + 3) inside |z|=2")
    print(f"  Number of zeros: {n4}  (expected {n})")
    assert n4 == n

    # --- Plots ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: roots of p1 with Rouche circle
    theta = np.linspace(0, 2*pi, 200)
    roots1_all = np.roots([1, 0, 0, 3, 0, 1])
    axes[0].plot(2*np.cos(theta), 2*np.sin(theta), 'b-', linewidth=1.5, label="|z|=2")
    axes[0].plot(np.cos(theta), np.sin(theta), 'g--', linewidth=1, alpha=0.5, label="|z|=1")
    axes[0].plot(roots1_all.real, roots1_all.imag, 'rx', markersize=8,
                 markeredgewidth=2, label="roots of $p$")
    axes[0].set_aspect('equal')
    axes[0].set_xlim(-2.5, 2.5)
    axes[0].set_ylim(-2.5, 2.5)
    axes[0].axhline(0, color='k', linewidth=0.3)
    axes[0].axvline(0, color='k', linewidth=0.3)
    axes[0].set_title("$z^5 + 3z^2 + 1$: 5 zeros inside $|z|=2$")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Right: argument of p1 on |z|=2 (winding number = 5)
    ts_plot = np.linspace(0, 2*pi, 500)
    zs = 2.0 * np.exp(1j * ts_plot)
    fs = p1(zs)
    unwrapped_arg = np.unwrap(np.angle(fs))
    axes[1].plot(ts_plot / pi, unwrapped_arg / pi, color="#1e77b4", linewidth=1.8)
    axes[1].set_xlabel("$t/\\pi$")
    axes[1].set_ylabel("arg $p(z(t))$ / $\\pi$")
    axes[1].set_title(f"Argument winding: total change = {int(round((unwrapped_arg[-1] - unwrapped_arg[0])/(2*pi)))} × 2π")
    axes[1].grid(True, alpha=0.4)

    fig.suptitle("Rouche's theorem: counting zeros via the argument principle", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "rouche_theorem.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
