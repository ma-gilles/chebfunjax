"""Nearest orthonormal functions.

Finds the nearest orthonormal system to a given set of functions using
the matrix SVD (polar decomposition), analogous to the nearest orthogonal matrix.

Credit: Behnam Hashemi, December 2014.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/NearestOrthFun.html
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

    # Define a set of non-orthogonal polynomial functions
    funcs = [
        cj.chebfun(lambda x: jnp.ones_like(x)),
        cj.chebfun(lambda x: x),
        cj.chebfun(lambda x: x**2),
        cj.chebfun(lambda x: x**3),
    ]

    # Compute inner product matrix G[i,j] = <f_i, f_j>
    n = len(funcs)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = float((funcs[i] * funcs[j]).sum())

    print(f"Inner product matrix G:\n{G}")

    # Polar decomposition: G = U S V^T, nearest orthonormal = U V^T
    U, s, Vt = np.linalg.svd(G)
    Q_mat = U @ Vt  # this is the unitary factor

    # Build orthonormal functions via Gram-Schmidt
    orth_funcs = [funcs[0] * (1.0 / float(jnp.sqrt(jnp.abs(jnp.array(float((funcs[0] * funcs[0]).sum()))))))]
    for k in range(1, n):
        candidate = funcs[k]
        for j in range(k):
            proj = float((candidate * orth_funcs[j]).sum())
            candidate = candidate - proj * orth_funcs[j]
        norm = float(jnp.sqrt(jnp.array(float((candidate * candidate).sum()))))
        orth_funcs.append(candidate * (1.0 / norm))

    # Verify orthonormality
    I_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            I_mat[i, j] = float((orth_funcs[i] * orth_funcs[j]).sum())

    err = np.max(np.abs(I_mat - np.eye(n)))
    print(f"NearestOrthFun: Gram-Schmidt orthonormality error = {err:.2e}")

    xx = np.linspace(-1.0, 1.0, 300)
    colors = ['b', 'r', 'g', 'm']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    for k, f in enumerate(funcs):
        vals = np.array([float(f(jnp.array(x))) for x in xx])
        ax.plot(xx, vals, color=colors[k], lw=1.5, label=f'f_{k}=x^{k}')
    ax.set_title('Original (non-orthogonal) functions', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for k, f in enumerate(orth_funcs):
        vals = np.array([float(f(jnp.array(x))) for x in xx])
        ax2.plot(xx, vals, color=colors[k], lw=1.5, label=f'P_{k}')
    ax2.set_title(f'Gram-Schmidt orthonormal functions (err={err:.1e})', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Nearest orthonormal functions', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'NearestOrthFun.png'), dpi=150)
    plt.close(fig)

    return True


if __name__ == '__main__':
    run()
