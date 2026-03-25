"""Orthogonal polynomials via Gram-Schmidt.

Demonstrates construction of orthogonal polynomials using the Gram-Schmidt
process (Stieltjes procedure) with chebfunjax.
Based on Chebfun example approx/OrthPolys.m by Nick Hale (June 2011).

Original: https://www.chebfun.org/examples/approx/OrthPolys.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj


def gram_schmidt_ortho(weight_fn, N, domain=(-1.0, 1.0)):
    """Compute N+1 orthonormal polynomials via Gram-Schmidt with given weight."""
    w = cj.chebfun(weight_fn, domain=domain)
    x = cj.chebfun(lambda t: t, domain=domain)

    # First polynomial: constant = 1/sqrt(integral of w)
    norm0 = float(jnp.sqrt((w * cj.chebfun(lambda t: jnp.ones_like(t), domain=domain)).sum()))
    P = [cj.chebfun(lambda t, n=norm0: jnp.ones_like(t) / n, domain=domain)]

    for k in range(1, N + 1):
        # New candidate: x * P[k-1]
        pk_new = x * P[k - 1]
        # Subtract off projections onto all previous P[j]
        for j in range(k):
            coeff = float((w * pk_new * P[j]).sum())
            pk_new = pk_new - coeff * P[j]
        # Normalize
        norm_sq = float((w * pk_new * pk_new).sum())
        norm_k = float(jnp.sqrt(jnp.abs(jnp.array(norm_sq))))
        pk_new = (1.0 / norm_k) * pk_new
        P.append(pk_new)

    return P


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/approx')
    os.makedirs(outdir, exist_ok=True)

    # Weight function w(x) = exp(pi*x) on [-1, 1]
    weight_fn = lambda x: jnp.exp(jnp.pi * x)
    N = 5
    P = gram_schmidt_ortho(weight_fn, N)

    # Verify orthonormality
    w = cj.chebfun(weight_fn)
    print("Orthonormality check (should be close to identity):")
    for i in range(N + 1):
        for j in range(N + 1):
            inn = float((w * P[i] * P[j]).sum())
            if abs(inn) > 1e-8:
                print(f"  <P_{i}, P_{j}> = {inn:.6f}", end="")
        if i == j:
            print(f"  <P_{i}, P_{i}> = {float((w*P[i]*P[i]).sum()):.6f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    xx = np.linspace(-1.0, 1.0, 500)
    colors = plt.cm.tab10(np.linspace(0, 0.6, N + 1))
    for k, pk in enumerate(P):
        vals = np.array(pk(jnp.array(xx)))
        ax.plot(xx, vals, color=colors[k], linewidth=1.6, label=f'$P_{k}$')

    ax.set_title(r'Orthonormal polynomials for $w(x) = e^{\pi x}$ on $[-1,1]$',
                 fontsize=12)
    ax.set_xlabel('$x$', fontsize=12)
    ax.legend(fontsize=10, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-4, 4)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'orthogonal_polynomials.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Also show standard Legendre-like: weight = 1
    P_leg = gram_schmidt_ortho(lambda x: jnp.ones_like(x), 5)
    w_leg = cj.chebfun(lambda x: jnp.ones_like(x))
    for i in range(4):
        inn = float((w_leg * P_leg[i] * P_leg[i]).sum())
        print(f"Legendre-like <P_{i},P_{i}> = {inn:.8f}  (should be 1.0)")
        assert abs(inn - 1.0) < 1e-8, f"Normalization failed for P_{i}: {inn}"

    print("orthogonal_polynomials: done")
    return True


if __name__ == "__main__":
    run()
