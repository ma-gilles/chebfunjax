"""Geometric Brownian motion.

Simulates geometric Brownian motion using random ODEs:
  y' = mu*y + sigma*f*y

where f is a smooth random function. As the wavelength lambda→0,
this approaches the Stratonovich SDE dX = mu*X dt + sigma*X ∘ dW.

Following ode-random/GBM.m by Nick Trefethen (May 2017).

Original MATLAB: https://www.chebfun.org/examples/ode-random/GBM.html
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
    print("Geometric Brownian motion")
    print("=" * 60)

    print("\nGBM ODE: y' = mu*y + sigma*f*y")
    print("Taking log: (log y)' = mu + sigma*f → no multiplicative noise")
    print("As lambda→0: Stratonovich SDE  dX = mu*X dt + sigma*X ∘ dW")

    domain = [0.0, 20.0]
    lam = 0.2
    n_paths = 5
    sigma = 1.0

    t_eval = np.linspace(0, 20, 400)
    dt = t_eval[1] - t_eval[0]

    def make_gbm_paths(mu, n_paths, seed_offset=0):
        """Compute GBM paths via log transform."""
        paths = []
        for k in range(n_paths):
            f_fn = cj.randnfun(lam, domain=domain, seed=k + seed_offset)
            f_vals = f_fn(t_eval)
            # Solve (log y)' = mu + sigma*f  → log y = mu*t + sigma * cumsum(f*dt)
            f_cumsum = np.cumsum(f_vals) * dt
            log_y = mu * t_eval + sigma * f_cumsum
            y = np.exp(log_y)  # y(0) = exp(0) = 1
            paths.append(y)
        return paths

    # Zero drift (mu=0): log y is a random walk, y has no bias on log scale
    print("\n1. Zero drift (mu=0, sigma=1): 5 paths")
    mu = 0.0
    paths_zero = make_gbm_paths(mu, n_paths, seed_offset=0)
    for k, y in enumerate(paths_zero):
        print(f"  Path {k+1}: y(20) = {y[-1]:.3f}")

    # Positive drift
    print("\n2. Positive drift (mu=0.2, sigma=1): 5 paths")
    mu = 0.2
    paths_pos = make_gbm_paths(mu, n_paths, seed_offset=0)
    mean_final_pos = np.mean([p[-1] for p in paths_pos])
    print(f"  Mean y(20) ≈ {mean_final_pos:.3f} (expected growth from mu={mu})")

    # Negative drift
    print("\n3. Negative drift (mu=-0.2, sigma=1): 5 paths")
    mu = -0.2
    paths_neg = make_gbm_paths(mu, n_paths, seed_offset=0)
    mean_final_neg = np.mean([p[-1] for p in paths_neg])
    print(f"  Mean y(20) ≈ {mean_final_neg:.4f} (expected decay from mu={mu})")

    # Check drift direction on average across multiple seeds
    mu_pos = 0.2
    mu_neg = -0.2
    n_check = 20
    final_pos = [make_gbm_paths(mu_pos, 1, seed_offset=k)[0][-1] for k in range(n_check)]
    final_neg = [make_gbm_paths(mu_neg, 1, seed_offset=k)[0][-1] for k in range(n_check)]

    mean_log_pos = np.mean(np.log(np.maximum(final_pos, 1e-30)))
    mean_log_neg = np.mean(np.log(np.maximum(final_neg, 1e-30)))
    print(f"\n  E[log y(20)] for mu=+0.2: {mean_log_pos:.2f} (expected: {0.2*20:.1f})")
    print(f"  E[log y(20)] for mu=-0.2: {mean_log_neg:.2f} (expected: {-0.2*20:.1f})")
    assert mean_log_pos > mean_log_neg, "Positive drift should give higher log-mean"
    print("\nPASS: positive drift shows upward trend in log-mean")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    colors = plt.cm.Set1(np.linspace(0, 1, n_paths))

    for i, y in enumerate(make_gbm_paths(0.0, n_paths, seed_offset=0)):
        axes[0].plot(t_eval, y, color=colors[i], linewidth=1.5, alpha=0.8)
    axes[0].set_title("GBM: zero drift (μ=0)", fontsize=11)
    axes[0].set_xlabel("t"); axes[0].set_ylabel("y")
    axes[0].grid(True, alpha=0.3); axes[0].set_ylim([0, None])

    for i, y in enumerate(make_gbm_paths(0.2, n_paths, seed_offset=0)):
        axes[1].plot(t_eval, y, color=colors[i], linewidth=1.5, alpha=0.8)
    axes[1].set_title("GBM: positive drift (μ=0.2)", fontsize=11)
    axes[1].set_xlabel("t"); axes[1].set_ylabel("y")
    axes[1].grid(True, alpha=0.3)

    for i, y in enumerate(make_gbm_paths(-0.2, n_paths, seed_offset=0)):
        axes[2].plot(t_eval, y, color=colors[i], linewidth=1.5, alpha=0.8)
    axes[2].set_title("GBM: negative drift (μ=-0.2)", fontsize=11)
    axes[2].set_xlabel("t"); axes[2].set_ylabel("y")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Geometric Brownian motion (σ=1, λ={lam})", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "gbm.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True


if __name__ == "__main__":
    run()
