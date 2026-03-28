"""Chebyshev interpolation of oscillatory entire functions.

For entire functions f(x) = sin(N*pi*x), Chebfun adapts the polynomial
degree proportional to N to achieve machine precision.

Credit: Mark Richardson, October 2011.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/Entire.html
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

    NN = [10, 110, 210, 310, 410, 510]

    lengths = []
    estimates = []
    for N in NN:
        ff = cj.chebfun(lambda x, N=N: jnp.sin(jnp.pi * N * x))
        lengths.append(len(ff))
        # Theoretical estimate: roughly 2N + a few
        estimates.append(2 * N + 2)

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    # Show some of the functions
    for N in [10, 50]:
        ff = cj.chebfun(lambda x, N=N: jnp.sin(jnp.pi * N * x))
        xx = np.linspace(-1.0, 1.0, max(600, 10 * N))
        vals = np.array([float(ff(jnp.array(x))) for x in xx])
        ax.plot(xx, vals, lw=1.2, label=f'N={N}')
    ax.set_title('sin(Nπx) for various N', fontsize=11)
    ax.legend(fontsize=9)
    ax2 = axes[1]
    ax2.scatter(NN, lengths, color='#0072BD', s=50, label='Chebfun length')
    ax2.scatter(NN, estimates, color='#D95319', marker='+', s=80, label='Estimate ≈ 2N')
    ax2.set_title('Degree vs. oscillation parameter N', fontsize=11)
    ax2.legend(fontsize=9)
    print("Entire:")
    print(f"{'N':>8}  {'estimate':>10}  {'chebfun len':>12}")
    for N, est, lv in zip(NN, estimates, lengths):
        print(f"  sin({N:>4}πx)  {est:>10}  {lv:>12}")

    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'Entire.png'), dpi=150)
    plt.close(fig)

    return True

if __name__ == '__main__':
    run()
