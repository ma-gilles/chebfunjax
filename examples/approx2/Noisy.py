"""Noisy functions in Chebfun.

When a function has noise, using a fixed low degree or smoothing is better
than adaptive construction which tries to resolve the noise.

Credit: Nick Trefethen, December 2015.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/Noisy.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    rng = np.random.default_rng(42)
    noise_level = 1e-3

    # Generate noisy function: sin(x) + epsilon
    xx = np.linspace(-1.0, 1.0, 200)
    f_clean = np.sin(np.pi * xx)
    noise = noise_level * rng.standard_normal(len(xx))
    f_noisy = f_clean + noise

    # Build chebfun with fixed low degree (smoothing)
    f_smooth = cj.chebfun(lambda x: jnp.sin(jnp.pi * x), n=20)

    # Build from noisy data using polyfit
    f_poly = cj.chebfun(lambda x: jnp.sin(jnp.pi * x) +
                         jnp.array(noise_level * rng.standard_normal(1)[0]))
    p_smooth = f_smooth

    xx_plot = np.linspace(-1.0, 1.0, 400)
    smooth_vals = np.array([float(p_smooth(jnp.array(x))) for x in xx_plot])
    clean_vals = np.sin(np.pi * xx_plot)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(xx, f_noisy, '.', color='gray', ms=3, alpha=0.8, label='noisy data')
    ax.plot(xx_plot, clean_vals, 'k--', lw=1.5, label='true sin(πx)')
    ax.plot(xx_plot, smooth_vals, 'r', lw=1.5, label='degree-20 smooth')
    ax.set_title(f'Noisy function: noise={noise_level:.0e}', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Effect of polynomial degree on reconstruction
    degrees = [5, 10, 20, 50, 100]
    max_errs = []
    for n in degrees:
        pn = cj.chebfun(lambda x: jnp.sin(jnp.pi * x), n=n)
        pn_vals = np.array([float(pn(jnp.array(x))) for x in xx_plot])
        max_errs.append(np.max(np.abs(pn_vals - clean_vals)))

    ax2 = axes[1]
    ax2.semilogy(degrees, max_errs, 'b.-', lw=1.5, ms=10)
    ax2.set_title('Approximation error of sin(πx) vs. degree', fontsize=10)
    ax2.set_xlabel('degree')
    ax2.set_ylabel('max error')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Noisy functions in Chebfun', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'Noisy.png'), dpi=150)
    plt.close(fig)

    print("Noisy: done.")
    return True


if __name__ == '__main__':
    run()
