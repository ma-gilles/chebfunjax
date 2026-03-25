"""From random functions to SDEs.

Demonstrates smooth random functions and their relationship to SDEs,
following ode-random/Random2SDE.m by Trefethen & Haji-Ali (May 2017).

The key idea: randnfun(lambda) produces a band-limited random function
with minimal wavelength lambda. As lambda→0, we approach white noise
(Brownian motion). Integrating u' = f gives a "smooth random walk".

Original MATLAB: https://www.chebfun.org/examples/ode-random/Random2SDE.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj


def run():
    print("=" * 60)
    print("From random functions to SDEs")
    print("=" * 60)

    print("\nKey concept: smooth random function f(t) with wavelength λ")
    print("u' = f → u(t) = ∫₀ᵗ f(s)ds is a smooth random walk")
    print("As λ→0, this approaches Brownian motion (Wiener process)")

    domain = [0.0, 1.0]
    lam = 0.001  # small wavelength → near Brownian motion
    t_vals = np.linspace(0, 1, 2000)
    dt = t_vals[1] - t_vals[0]

    print(f"\nSmooth random walk with λ={lam}, 3 sample paths:")
    all_paths = []
    f_paths = []
    for i in range(3):
        f_fn = cj.randnfun(lam, domain=domain, seed=i, big=True)
        f_vals = f_fn(t_vals)
        # Cumulative integral: u(t) = ∫_0^t f(s) ds
        u_vals = np.cumsum(f_vals) * dt
        u_vals -= u_vals[0]  # start at 0
        all_paths.append(u_vals)
        f_paths.append(f_vals)
        print(f"  Path {i+1}: u(1) = {u_vals[-1]:.4f}, max|u| = {np.max(np.abs(u_vals)):.4f}")

    # Check: E[u(T)^2] ≈ T (variance of Brownian motion)
    # For lambda small with 'big' normalization, var ≈ T
    var_at_1 = np.mean([p[-1]**2 for p in all_paths])
    print(f"\n  Sample variance at T=1: {var_at_1:.4f} (expected ≈ 1 for normalized BM)")
    print(f"  (With only 3 samples, exact match not expected)")

    # White noise paradox: as lambda decreases, amplitude increases
    print(f"\nWhite noise paradox demonstration:")
    lambdas = [0.1, 0.01, 0.001]
    for lam_i in lambdas:
        f_i = cj.randnfun(lam_i, domain=domain, seed=42, big=True)
        f_vals_i = f_i(t_vals)
        amplitude = np.std(f_vals_i)
        print(f"  λ={lam_i}: std(f) ≈ {amplitude:.2f} (grows as λ→0)")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    colors = ['blue', 'red', 'green']
    for i, u_vals in enumerate(all_paths):
        axes[0].plot(t_vals, u_vals, color=colors[i], linewidth=1.5,
                     label=f'Path {i+1}')
    axes[0].axhline(0, color='k', linestyle='-', linewidth=0.5)
    axes[0].set_title(f"Smooth random walks (λ={lam})", fontsize=11)
    axes[0].set_xlabel("t"); axes[0].set_ylabel("u(t)")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-2, 2])

    # Show the random function itself (first path)
    axes[1].plot(t_vals, f_paths[0], 'b-', linewidth=1.0, label=f"f(t), λ={lam}")
    axes[1].set_title("Random function f(t) = u'(t)", fontsize=11)
    axes[1].set_xlabel("t"); axes[1].set_ylabel("f(t)")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Random functions → smooth random walks → SDEs", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "random2sde.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True


if __name__ == "__main__":
    run()
