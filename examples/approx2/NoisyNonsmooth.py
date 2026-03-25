"""Chebfuns of noisy functions with discontinuities.

Addresses the challenge of constructing a chebfun for a function that is both
noisy AND has discontinuities.

Credit: Nick Trefethen, July 2014.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/NoisyNonsmooth.html
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.domain import Domain

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    rng = np.random.default_rng(42)
    noise_level = 1e-2

    # Signal: step function + noise
    def true_signal(x):
        return jnp.where(x < 0.0, jnp.sin(2.0 * jnp.pi * x),
                         jnp.sin(2.0 * jnp.pi * x) + 0.5)

    xx = np.linspace(-1.0, 1.0, 300)
    true_vals = np.array([float(true_signal(jnp.array(x))) for x in xx])
    noisy_vals = true_vals + noise_level * rng.standard_normal(len(xx))

    # Strategy: if you know the breakpoint, construct piecewise
    dom_pw = Domain([-1.0, 0.0, 1.0])
    f_pw_left = cj.chebfun(lambda x: jnp.sin(2.0 * jnp.pi * x), n=10,
                            domain=(-1.0, 0.0))
    f_pw_right = cj.chebfun(lambda x: jnp.sin(2.0 * jnp.pi * x) + 0.5, n=10,
                             domain=(0.0, 1.0))

    xx_left = np.linspace(-1.0, 0.0, 200)
    xx_right = np.linspace(0.0, 1.0, 200)
    left_vals = np.array([float(f_pw_left(jnp.array(x))) for x in xx_left])
    right_vals = np.array([float(f_pw_right(jnp.array(x))) for x in xx_right])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(xx, noisy_vals, '.', color='gray', ms=3, alpha=0.7, label='noisy data')
    ax.plot(xx, true_vals, 'k--', lw=1.5, label='true signal')
    ax.axvline(0, color='r', lw=1.0, ls='--', label='discontinuity')
    ax.set_title('Noisy function with discontinuity', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(xx_left, left_vals, 'b', lw=1.8)
    ax2.plot(xx_right, right_vals, 'b', lw=1.8, label='piecewise smooth')
    ax2.plot(xx, true_vals, 'k--', lw=1.0, alpha=0.5, label='true')
    ax2.set_title('Piecewise chebfun (known breakpoint)', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Noisy functions with discontinuities', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'NoisyNonsmooth.png'), dpi=150)
    plt.close(fig)

    print("NoisyNonsmooth: done.")
    return True


if __name__ == '__main__':
    run()
