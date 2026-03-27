"""Orthogonal polynomials via the Lanczos process.

Uses the Lanczos process (three-term recurrence) to construct orthogonal
polynomials with respect to a given weight function.

Credit: Pedro Gonnet, November 2011.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/OrthPolysLanczos.html
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
from chebfunjax.plotting import chebfun_style
chebfun_style()


_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def lanczos_orth(w_func, N):
    """Construct N orthonormal polynomials via Lanczos/Stieltjes process."""
    w = cj.chebfun(w_func)
    x = cj.chebfun(lambda t: t)

    polys = []
    betas = []

    # P_0 = 1/sqrt(<w, 1>)
    norm0 = float(jnp.sqrt(jnp.array(float(w.sum()))))
    p0 = cj.chebfun(lambda t: jnp.ones_like(t) / norm0)
    polys.append(p0)

    for k in range(1, N + 1):
        # alpha_k = <w * x * P_{k-1}, P_{k-1}>
        xpk = x * polys[k - 1]
        alpha = float((w * xpk * polys[k - 1]).sum())

        if k == 1:
            pk_unnorm = xpk - alpha * polys[k - 1]
        else:
            pk_unnorm = xpk - alpha * polys[k - 1] - float(betas[k - 2]) * polys[k - 2]

        beta = float(jnp.sqrt(jnp.array(float((w * pk_unnorm * pk_unnorm).sum()))))
        betas.append(beta)
        polys.append(pk_unnorm * (1.0 / beta))

    return polys, betas


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    N = 5
    def w_func(x): return jnp.exp(jnp.pi * x)
    polys, betas = lanczos_orth(w_func, N)

    print(f"Lanczos beta coefficients: {[f'{b:.4f}' for b in betas]}")

    # Verify orthonormality
    w = cj.chebfun(w_func)
    I_mat = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            I_mat[i, j] = float((w * polys[i] * polys[j]).sum())
    err = np.max(np.abs(I_mat - np.eye(N + 1)))
    print(f"Lanczos orthonormality error = {err:.2e}")

    xx = np.linspace(-1.0, 1.0, 300)
    colors = ['b', 'r', 'g', 'm', 'orange', 'c']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    for k, p in enumerate(polys[:min(6, N+1)]):
        vals = np.array([float(p(jnp.array(x))) for x in xx])
        ax.plot(xx, vals, color=colors[k % len(colors)], lw=1.5, label=f'P_{k}')
    ax.set_title('Lanczos orthogonal polynomials wrt exp(πx)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    im = ax2.imshow(I_mat, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax2)
    ax2.set_title(f'Inner product matrix (err={err:.1e})', fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'OrthPolysLanczos.png'), dpi=150)
    plt.close(fig)

    return True


if __name__ == '__main__':
    run()
