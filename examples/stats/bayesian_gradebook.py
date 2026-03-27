"""Bayesian gradebook.

Demonstrates Bayesian updating of student ability estimates from
assessment scores, comparing posterior mode/mean to traditional
running averages. Translated from stats/BayesianGradebook.m.

Original: https://www.chebfun.org/examples/stats/BayesianGradebook.html
Author: Toby Driscoll, November 2013
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()



def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    # Ability parameter theta in [0, 1]
    n_theta = 500
    theta = np.linspace(0, 1, n_theta)
    dtheta = theta[1] - theta[0]

    # Prior: truncated normal centered at 0.7, large variance
    prior_unnorm = np.exp(-((theta - 0.7) / 0.3)**2 / 2)
    prior = prior_unnorm / np.trapezoid(prior_unnorm, theta)

    # Likelihood model: truncated normal with sigma=0.06
    sigma = 0.06
    q = np.array([np.trapezoid(np.exp(-((theta - t) / sigma)**2 / 2), theta)
                  for t in theta])

    def likelihood(x, theta):
        return np.exp(-((theta - x) / sigma)**2 / 2) / q

    def E_theta(prob):
        return np.trapezoid(theta * prob, theta)

    def mode_theta(prob):
        return theta[np.argmax(prob)]

    def bayes_update(scores, prior):
        belief = prior.copy()
        trad_avgs = []
        means = []
        modes = []
        stds = []
        beliefs = [belief.copy()]

        for k, x in enumerate(scores):
            b = belief * likelihood(x, theta)
            b /= np.trapezoid(b, theta)
            belief = b
            beliefs.append(b.copy())
            trad_avgs.append(np.mean(scores[:k+1]))
            means.append(E_theta(b))
            modes.append(mode_theta(b))
            var = np.trapezoid((theta - means[-1])**2 * b, theta)
            stds.append(np.sqrt(var))

        return beliefs, trad_avgs, means, modes, stds

    fig, axes = plt.subplots(1, 3)

    # --- 1. Poor student ---
    scores_poor = [0.55, 0.67, 0.62, 0.66]
    beliefs_poor, trad_poor, means_poor, modes_poor, stds_poor = bayes_update(scores_poor, prior)

    print("Poor student:")
    print(f"{'Method':<20} {'m-3':>7} {'m-2':>7} {'m-1':>7} {'m':>7}")
    print("-" * 50)
    m = len(scores_poor)
    print(f"{'Traditional':<20} " + " ".join(f"{v:7.3f}" for v in trad_poor[-4:]))
    print(f"{'Bayes Mode':<20} " + " ".join(f"{v:7.3f}" for v in modes_poor[-4:]))
    print(f"{'Bayes Mean':<20} " + " ".join(f"{v:7.3f}" for v in means_poor[-4:]))
    print(f"{'Std dev':<20} " + " ".join(f"{v:7.3f}" for v in stds_poor[-4:]))

    colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(beliefs_poor)))
    for i, bel in enumerate(beliefs_poor):
        axes[0].plot(theta, bel, color=colors[i], linewidth=1.5 + i * 0.3)
    axes[0].set_title('Poor student (posterior evolution)', fontsize=10)
    axes[0].set_xlabel('θ'); axes[0].set_ylabel('P(θ|x)')
    axes[0].grid(True, alpha=0.3)

    # --- 2. Good student ---
    scores_good = [s + 0.3 for s in scores_poor]
    print("\nGood student:")
    beliefs_good, trad_good, means_good, modes_good, stds_good = bayes_update(scores_good, prior)
    print(f"{'Method':<20} " + " ".join([f"{'score':>7}"] * len(scores_good)))
    print(f"{'Traditional':<20} " + " ".join(f"{v:7.3f}" for v in trad_good))
    print(f"{'Bayes Mean':<20} " + " ".join(f"{v:7.3f}" for v in means_good))

    colors2 = plt.cm.Reds(np.linspace(0.3, 1.0, len(beliefs_good)))
    for i, bel in enumerate(beliefs_good):
        axes[1].plot(theta, bel, color=colors2[i], linewidth=1.5 + i * 0.3)
    axes[1].set_title('Good student (boundary effect)', fontsize=10)
    axes[1].set_xlabel('θ'); axes[1].set_ylabel('P(θ|x)')
    axes[1].grid(True, alpha=0.3)

    # --- 3. Many assessments, wide sigma ---
    sigma = 0.15
    q2 = np.array([np.trapezoid(np.exp(-((theta - t) / sigma)**2 / 2), theta)
                   for t in theta])
    def likelihood2(x, theta):
        return np.exp(-((theta - x) / sigma)**2 / 2) / q2

    scores_many = [0.88, 0.90, 0.46, 0.86, 0.93, 0.61, 0.95, 0.89, 0.84, 0.76]

    def bayes_update2(scores, prior):
        belief = prior.copy()
        means = []; stds = []; trad = []
        for k, x in enumerate(scores):
            b = belief * likelihood2(x, theta)
            b /= np.trapezoid(b, theta)
            belief = b
            m_val = np.trapezoid(theta * b, theta)
            means.append(m_val)
            var = np.trapezoid((theta - m_val)**2 * b, theta)
            stds.append(np.sqrt(var))
            trad.append(np.mean(scores[:k+1]))
        return means, stds, trad

    means_m, stds_m, trad_m = bayes_update2(scores_many, prior)
    k_vals = np.arange(1, len(scores_many) + 1)
    axes[2].plot(k_vals, trad_m, 'k.-', markersize=10, linewidth=2, label='Traditional avg')
    axes[2].plot(k_vals, means_m, 'r.-', markersize=10, linewidth=2, label='Bayes mean')
    axes[2].fill_between(k_vals,
                         np.array(means_m) - np.array(stds_m),
                         np.array(means_m) + np.array(stds_m),
                         alpha=0.2, color='red', label='±1 std')
    axes[2].set_title('Inconsistent student (σ=0.15)', fontsize=10)
    axes[2].set_xlabel('Assessment #'); axes[2].set_ylabel('Ability estimate')
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)

    print("\nInconsistent student:")
    print(f"{'Method':<20} " + " ".join(f"{v:5.3f}" for v in trad_m[-4:]))
    print(f"{'Bayes Mean':<20} " + " ".join(f"{v:5.3f}" for v in means_m[-4:]))
    print(f"{'Std dev':<20} " + " ".join(f"{v:5.3f}" for v in stds_m[-4:]))

    fig.suptitle('Bayesian Gradebook: Estimating Student Ability', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'bayesian_gradebook.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("bayesian_gradebook: done")
    return True


if __name__ == "__main__":
    run()
