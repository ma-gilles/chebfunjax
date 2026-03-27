"""Orthogonal polynomials via the Gram-Schmidt process.

Constructs orthonormal polynomials with respect to a non-standard weight
w(x) = exp(pi*x) on [-1,1] using the Gram-Schmidt (Stieltjes) process.

Credit: Nick Hale, June 2011.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/OrthPolys.html
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

def orth_poly_gram_schmidt(w_func, N):
    """Build N+1 orthonormal polynomials wrt weight w via Gram-Schmidt."""
    w = cj.chebfun(w_func)
    x = cj.chebfun(lambda t: t)

    # Normalize the constant polynomial
    norm0 = float(jnp.sqrt((w * cj.chebfun(lambda t: jnp.ones_like(t))).sum()))
    polys = [cj.chebfun(lambda t, n=norm0: jnp.ones_like(t) / n)]

    for k in range(1, N + 1):
        # New candidate: x * P_{k-1}
        xpk = x * polys[k - 1]
        candidate = xpk
        # Subtract projections
        for j in range(k):
            c = float((w * xpk * polys[j]).sum())
            candidate = candidate - c * polys[j]
        # Normalize
        norm_sq = float((w * candidate * candidate).sum())
        polys.append(candidate * (1.0 / float(jnp.sqrt(jnp.array(norm_sq)))))

    return polys

def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    N = 4
    def w_func(x): return jnp.exp(jnp.pi * x)
    polys = orth_poly_gram_schmidt(w_func, N)

    xx = np.linspace(-1.0, 1.0, 400)
    colors = ['b', 'r', 'g', 'm', 'orange']

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    for k, p in enumerate(polys):
        p_vals = np.array([float(p(jnp.array(x))) for x in xx])
        ax.plot(xx, p_vals, color=colors[k % len(colors)], lw=1.5,
                label=f'P_{k}')
    ax.set_title('Orthogonal polynomials wrt w=exp(πx)', fontsize=11)
    ax.legend(fontsize=9)
    # Verify orthonormality: inner product matrix should be I
    w = cj.chebfun(w_func)
    I_matrix = np.zeros((N + 1, N + 1))
    for i, pi in enumerate(polys):
        for j, pj in enumerate(polys):
            I_matrix[i, j] = float((w * pi * pj).sum())
    err = np.max(np.abs(I_matrix - np.eye(N + 1)))
    print(f"OrthPolys: orthonormality error = {err:.2e}")

    ax2 = axes[1]
    im = ax2.imshow(I_matrix, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax2)
    ax2.set_title(f'Inner product matrix (should be I, err={err:.1e})', fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'OrthPolys.png'), dpi=150)
    plt.close(fig)

    return True

if __name__ == '__main__':
    run()
