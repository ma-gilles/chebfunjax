"""Local complexity of a function.

Demonstrates how the local number of Chebyshev coefficients needed reflects
the local complexity (oscillation) of the function.

Credit: Nick Trefethen, June 2011.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/Local.html
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

    # Function with varying local complexity: fast oscillation on left, slow on right
    def f_func(x):
        return jnp.sin(x * (20.0 - 15.0 * x))  # increasing frequency

    # Build with splitting to adapt locally (pass breakpoints as list)
    breakpoints = [-1.0, -0.5, 0.0, 0.5, 1.0]
    f = cj.chebfun(f_func, domain=breakpoints)

    xx = np.linspace(-1.0, 1.0, 800)
    f_vals = np.array([float(f(jnp.array(x))) for x in xx])
    f_exact = np.array([float(f_func(jnp.array(x))) for x in xx])

    # Show local lengths per piece (f.funs are _Piece objects with .tech.coeffs)
    piece_lengths = [len(fun.tech.coeffs) for fun in f.funs]
    piece_midpoints = [0.5 * (bp[0] + bp[1])
                       for bp in zip(breakpoints[:-1], breakpoints[1:])]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(xx, f_vals, 'b', lw=1.5)
    for bp in breakpoints[1:-1]:
        ax.axvline(bp, color='r', lw=0.8, ls='--')
    ax.set_title('f(x) = sin(x(20−15x)) with breakpoints', fontsize=10)
    ax.set_xlabel('x')
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(piece_midpoints, piece_lengths, width=0.4, color='b', alpha=0.7)
    ax2.set_title('Chebyshev coefficients per piece (local complexity)', fontsize=10)
    ax2.set_xlabel('piece midpoint')
    ax2.set_ylabel('number of coefficients')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Local complexity of a function', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'Local.png'), dpi=150)
    plt.close(fig)

    print(f"Local: piece lengths = {piece_lengths}")
    return True


if __name__ == '__main__':
    run()
