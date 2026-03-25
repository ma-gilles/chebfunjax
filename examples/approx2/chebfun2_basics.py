"""Chebfun2 basics: smooth 2D function approximation.

Demonstrates constructing 2D Chebfun approximations, evaluating at points,
integrating, and differentiating. Based on Chebfun2 examples.

Original: https://www.chebfun.org/examples/approx2/
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


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/approx2')
    os.makedirs(outdir, exist_ok=True)

    # --- Basic 2D function: f(x,y) = exp(-(x^2 + y^2)) ----------------
    f = cj.chebfun2(lambda x, y: jnp.exp(-(x**2 + y**2)))
    print(f"Gaussian f: shape = {f.shape if hasattr(f, 'shape') else 'N/A'}")

    # Evaluate at a point
    val = float(f(jnp.array(0.5), jnp.array(0.3)))
    exact = float(jnp.exp(-(0.5**2 + 0.3**2)))
    print(f"f(0.5, 0.3) = {val:.12f}  (exact: {exact:.12f})")
    assert abs(val - exact) < 1e-10

    # 2D integral: integral of exp(-(x^2+y^2)) over [-1,1]^2
    integral = float(f.sum())
    # Reference: pi * erf(1)^2 ≈ 2.230985...
    import scipy.special
    exact_int = float(np.pi) * scipy.special.erf(1.0)**2
    print(f"∬ f dA = {integral:.10f}  (exact: {exact_int:.10f})")
    assert abs(integral - exact_int) < 1e-8

    # --- Plot: contour and surface plots --------------------------------
    xx = np.linspace(-1, 1, 100)
    yy = np.linspace(-1, 1, 100)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = np.exp(-(XX**2 + YY**2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Contour
    cs = axes[0].contourf(XX, YY, ZZ, levels=20, cmap='Blues')
    plt.colorbar(cs, ax=axes[0])
    axes[0].set_title(r'$f(x,y) = e^{-(x^2+y^2)}$', fontsize=11)
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$y$')
    axes[0].set_aspect('equal')

    # 3D surface
    ax3d = fig.add_subplot(122, projection='3d')  # This replaces axes[1]
    axes[1].remove()
    ax3d.plot_surface(XX, YY, ZZ, cmap='Blues', alpha=0.8)
    ax3d.set_title(r'$e^{-(x^2+y^2)}$', fontsize=11)
    ax3d.set_xlabel('$x$')
    ax3d.set_ylabel('$y$')
    ax3d.set_zlabel('$f$')

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'chebfun2_basics.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Second function: f = sin(pi*x)*cos(pi*y) ----------------------
    g = cj.chebfun2(lambda x, y: jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y))
    # Integral over [-1,1]^2 should be 0 (odd in x)
    integral_g = float(g.sum())
    print(f"∬ sin(πx)cos(πy) dA = {integral_g:.2e}  (exact: 0)")
    assert abs(integral_g) < 1e-12

    print("chebfun2_basics: done")
    return True


if __name__ == "__main__":
    run()
