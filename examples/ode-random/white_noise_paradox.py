"""The white noise paradox.

Demonstrates how band-limited random functions with decreasing wavelength lambda
have increasing amplitude, approaching infinite amplitude in the limit lambda→0
(the white noise paradox). Connects to the ultraviolet catastrophe in physics.

Following ode-random/WhiteNoiseParadox.m by Nick Trefethen (May 2017).

Original MATLAB: https://www.chebfun.org/examples/ode-random/WhiteNoiseParadox.html
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



def run():
    print("=" * 60)
    print("The white noise paradox")
    print("=" * 60)

    print("\nKey insight: as lambda→0, random function amplitude → ∞")
    print("(Same as the ultraviolet catastrophe in physics)")
    print("\nBand-limited random function with wavenumber cutoff ~2π/lambda")
    print("Amplitude grows like lambda^(-1/2) (the 'big' normalization)")

    domain = [0.0, 1.0]
    t_eval = np.linspace(0, 1, 2000)

    # Three lambdas: 1/4, 1/16, 1/64
    lambdas = [1/4, 1/16, 1/64]
    lambda_names = ['1/4', '1/16', '1/64']

    paths = []
    stats = []
    for i, lam in enumerate(lambdas):
        f_fn = cj.randnfun(lam, domain=domain, seed=1, big=True)
        f_vals = np.array([float(f_fn(np.array(ti))) for ti in t_eval])
        paths.append(f_vals)

        std_f = np.std(f_vals)
        max_f = np.max(np.abs(f_vals))
        stats.append((std_f, max_f))
        print(f"\n  lambda={lambda_names[i]}:")
        print(f"    std(f) = {std_f:.2f}")
        print(f"    max|f| = {max_f:.2f}")
        # Expected: std ~ (lambda/2)^(-1/2) * normalized_std
        expected_growth = 1.0 / np.sqrt(lam / 0.25)
        print(f"    Amplitude ratio vs lambda=1/4: {std_f/stats[0][0]:.2f}"
              f" (expected ~{expected_growth:.2f})")

    # Verify amplitude grows as lambda decreases (use multiple seeds for robustness)
    # Average over a few seeds to get reliable statistics
    t_check = np.linspace(0, 1, 2000)
    stds = []
    for lam_check in lambdas:
        std_avg = 0.0
        for s in range(5):
            fi = cj.randnfun(lam_check, domain=domain, seed=s, big=True)
            fv = fi(t_check)
            std_avg += np.std(fv)
        stds.append(std_avg / 5)

    print(f"\n  Average std over 5 seeds: {stds}")
    assert stds[1] > stds[0], \
        f"Mean amplitude should increase as lambda decreases: {stds}"
    print("\n  PASS: amplitude grows as lambda decreases (white noise paradox)")

    # Also show the resolution: Brownian motion (integral of f) is bounded
    print("\nResolution: integral of f (Brownian-like path) stays bounded")
    for i, (lam, f_vals) in enumerate(zip(lambdas, paths)):
        u_vals = np.cumsum(f_vals) * (t_eval[1] - t_eval[0])
        u_vals -= u_vals[0]  # start at 0
        print(f"  lambda={lambda_names[i]}: "
              f"max|u| = {np.max(np.abs(u_vals)):.3f}, "
              f"u(1) = {u_vals[-1]:.3f}")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 3)

    for i, (lam, f_vals) in enumerate(zip(lambdas, paths)):
        axes[i].plot(t_eval, f_vals, linewidth=max(0.5, 1.2 - 0.3*i),
                     color='steelblue', alpha=0.8)
        axes[i].set_title(f"lambda = {lambda_names[i]}", fontsize=11)
        axes[i].set_xlabel("t"); axes[i].set_ylabel("f(t)")
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim([-30, 30])
        axes[i].text(0.05, 0.92, f"std = {stats[i][0]:.1f}",
                     transform=axes[i].transAxes, fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle("White noise paradox: amplitude grows as λ→0", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "white_noise_paradox.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True


if __name__ == "__main__":
    run()
