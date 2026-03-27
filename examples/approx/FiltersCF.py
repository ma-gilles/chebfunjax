"""Digital filters via CF approximation.

CF (Caratheodory-Fejer) approximation is useful for designing digital
filters. This example illustrates polynomial approximation of ideal
low-pass filter characteristics.

Credit: Nick Trefethen, April 2014.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/FiltersCF.html
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

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')

def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # Ideal low-pass filter: step function at w=0.5
    def lowpass(x): return jnp.where(jnp.abs(x) < 0.5, 1.0, 0.0)

    # Approximate the step with a smooth sigmoid
    def smooth_lowpass(x, beta=20.0):
        return 1.0 / (1.0 + jnp.exp(beta * (jnp.abs(x) - 0.5)))

    f_step = cj.chebfun(lambda x: jnp.where(jnp.abs(x) < 0.5, 1.0, 0.0),
                        domain=(-1.0, -0.5, 0.5, 1.0))
    f_smooth = cj.chebfun(lambda x: smooth_lowpass(x, beta=20.0))

    xx = np.linspace(-1.0, 1.0, 600)
    step_vals = np.where(np.abs(xx) < 0.5, 1.0, 0.0)
    smooth_vals = np.array([float(f_smooth(jnp.array(x))) for x in xx])

    # Polynomial approximations of increasing degree
    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    ax.plot(xx, step_vals, 'k--', lw=1.5, label='ideal step')
    for n, col in [(5, 'b'), (10, 'r'), (20, 'g')]:
        pn = f_smooth.polyfit(n)
        pn_vals = np.array([float(pn(jnp.array(x))) for x in xx])
        ax.plot(xx, pn_vals, '-', color=col, lw=1.3, label=f'deg {n}')
    ax.set_title('Low-pass filter approximation by polynomials', fontsize=10)
    ax.legend(fontsize=8)
    # Gibbs phenomenon: best polynomial approx of step function
    ax2 = axes[1]
    ax2.plot(xx, step_vals, 'k--', lw=1.5, label='ideal')
    for n, col in [(10, 'b'), (30, 'r'), (60, 'g')]:
        pn = f_step.polyfit(n)
        pn_vals = np.array([float(pn(jnp.array(x))) for x in xx])
        ax2.plot(xx, pn_vals, '-', color=col, lw=1.2, alpha=0.8, label=f'deg {n}')
    ax2.set_title('Gibbs phenomenon: step function approx', fontsize=10)
    ax2.legend(fontsize=8)
    fig.suptitle('Digital filter design via polynomial approximation', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'FiltersCF.png'), dpi=150)
    plt.close(fig)

    print("FiltersCF: done.")
    return True

if __name__ == '__main__':
    run()
