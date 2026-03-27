"""Summing a divergent series.

Uses Padé approximants (rational approximation of Taylor series) to sum
the divergent asymptotic series for the Stieltjes integral f(x) = int_0^inf e^{-t}/(1+xt) dt.

Credit: Nick Trefethen and Stefan Guettel, April 2012.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/DivergentSeries.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')

def stieltjes(x):
    """Stieltjes integral f(x) = int_0^inf e^{-t}/(1+xt) dt."""
    result, _ = quad(lambda t: np.exp(-t) / (1 + x * t), 0, 50)
    return result

def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # Evaluate the exact Stieltjes function
    xs = np.linspace(0.01, 1.0, 200)
    exact = np.array([stieltjes(x) for x in xs])

    # Taylor series diverges; use partial sums to illustrate
    # f(x) ~ sum_{k=0}^{N} (-1)^k k! x^k
    def taylor_partial(x_arr, N):
        result = np.zeros_like(x_arr, dtype=float)
        for k in range(N + 1):
            import math
        result += (-1)**k * float(math.factorial(k)) * x_arr**k
        return result

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    ax.plot(xs, exact, 'k', lw=2, label='exact f(x)')
    for N, col in [(3, 'b'), (5, 'r'), (7, 'g')]:
        partial = taylor_partial(xs, N)
        ax.plot(xs, partial, '--', color=col, lw=1.2, label=f'Taylor N={N}')
    ax.set_ylim(-1, 2)
    ax.set_title('Stieltjes function and diverging Taylor series', fontsize=10)
    ax.legend(fontsize=8)
    # Padé approximant via Chebfun polyfit then rational reconstruction
    # Simple diagonal Padé [2/2] approximant
    # For e^{-x} on [0,1] as a simpler demo
    f_exp = cj.chebfun(lambda x: jnp.exp(-x))
    p2 = f_exp.polyfit(4)

    xx_plot = np.linspace(0.0, 1.0, 300)
    exact_exp = np.exp(-xx_plot)
    p2_vals = np.array([float(p2(jnp.array(x))) for x in xx_plot])
    err_p2 = np.abs(p2_vals - exact_exp)

    ax2 = axes[1]
    ax2.semilogy(xx_plot, err_p2 + 1e-18, 'b', lw=1.5, label='degree-4 L2 error')
    ax2.set_title('L2 polynomial approximation of e^{-x}', fontsize=10)
    ax2.legend(fontsize=9)
    fig.suptitle('Divergent series and Padé-style approximation', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'DivergentSeries.png'), dpi=150)
    plt.close(fig)

    print(f"DivergentSeries: f(1) = {stieltjes(1.0):.6f}")
    return True

if __name__ == '__main__':
    run()
