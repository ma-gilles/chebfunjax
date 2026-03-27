"""Vector calculus identities and Chebfun2v.

Demonstrates grad, div, curl, and Laplacian using Chebfun2v,
following veccalc/CheckingVectorCalculus.m by Trefethen (2010).

curl(grad(f)) = 0
div(curl(F)) = 0
div(grad(f)) = Δf

Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
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

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

def run():
    print("=" * 60)
    print("Vector calculus identities with Chebfun2v")
    print("=" * 60)

    # Scalar potential
    f = chebfun2(lambda x, y: jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y))
    print(f"\nf = sin(πx)cos(πy), rank {f.rank}")

    # Gradient: Fx = df/dx = pi*cos(pi*x)*cos(pi*y)
    #           Fy = df/dy = -pi*sin(pi*x)*sin(pi*y)
    # Note: dim=1 gives df/dy, dim=2 gives df/dx in this implementation
    Fx = f.diff(dim=2)   # d/dx
    Fy = f.diff(dim=1)   # d/dy
    F = Chebfun2v.from_functions(Fx, Fy)
    print(f"grad(f) components: ({Fx.rank}, {Fy.rank}) ranks")

    # curl(grad(f)) = dFy/dx - dFx/dy = 0 (exactly)
    curl_grad_f = F.curl()
    print(f"\ncurl(grad(f)) norm: {abs(float(curl_grad_f.norm())):.2e}  (exact: 0)")
    assert abs(float(curl_grad_f.norm())) < 1e-6

    # div(grad(f)) = Laplacian(f) = -2*pi^2 * sin(pi*x)*cos(pi*y)
    div_grad_f = F.div()
    # Verify at a test point
    x_test, y_test = 0.3, 0.4
    val_lap = float(div_grad_f(jnp.array(x_test), jnp.array(y_test)))
    exact_lap = float(-2 * jnp.pi**2 * jnp.sin(jnp.pi * x_test) * jnp.cos(jnp.pi * y_test))
    err_lap = abs(val_lap - exact_lap)
    print(f"\ndiv(grad(f))({x_test},{y_test}) = {val_lap:.8f}")
    print(f"Δf exact           = {exact_lap:.8f}")
    print(f"Error              = {err_lap:.2e}")
    assert err_lap < 1e-6

    # --- curl(G) for arbitrary G ---
    # G = (x*sin(y), cos(x)*y)
    # curl(G) = dGy/dx - dGx/dy = -sin(x)*y - x*cos(y)
    Gx = chebfun2(lambda x, y: x * jnp.sin(y))
    Gy = chebfun2(lambda x, y: jnp.cos(x) * y)
    G = Chebfun2v.from_functions(Gx, Gy)
    print(f"\nG = (x sin(y), cos(x) y)")

    curl_G = G.curl()
    val_curl = float(curl_G(jnp.array(0.5), jnp.array(0.3)))
    exact_curl = float(-jnp.sin(jnp.array(0.5)) * 0.3 - 0.5 * jnp.cos(jnp.array(0.3)))
    print(f"curl(G)(0.5, 0.3) = {val_curl:.8f}  (exact: {exact_curl:.8f})")
    err_curl = abs(val_curl - exact_curl)
    assert err_curl < 1e-8, f"curl error: {err_curl}"

    # div(G) = dGx/dx + dGy/dy = sin(y) + cos(x)
    div_G = G.div()
    val_div = float(div_G(jnp.array(0.5), jnp.array(0.3)))
    exact_div = float(jnp.sin(jnp.array(0.3)) + jnp.cos(jnp.array(0.5)))
    print(f"div(G)(0.5, 0.3) = {val_div:.8f}  (exact: {exact_div:.8f})")
    assert abs(val_div - exact_div) < 1e-8

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    xs = np.linspace(-1, 1, 40)
    ys = np.linspace(-1, 1, 40)
    X, Y = np.meshgrid(xs, ys)

    # Gradient field
    U = np.pi * np.cos(np.pi * X) * np.cos(np.pi * Y)
    V = -np.pi * np.sin(np.pi * X) * np.sin(np.pi * Y)
    speed = np.sqrt(U**2 + V**2)

    axes[0].streamplot(xs, ys, U.T, V.T, color=speed.T, cmap='viridis',
                        linewidth=1, density=1)
    axes[0].contourf(X, Y, np.sin(np.pi * X) * np.cos(np.pi * Y),
                      levels=20, alpha=0.3, cmap='RdBu_r')
    axes[0].set_title("grad(sin(πx)cos(πy))", fontsize=12)

    # curl_G plot
    curl_exact = -np.sin(X) * Y - X * np.cos(Y)
    im = axes[1].contourf(X, Y, curl_exact, levels=20, cmap="RdBu_r")
    axes[1].set_title("curl(x·sin(y), cos(x)·y)", fontsize=12)
    fig.colorbar(im, ax=axes[1])

    fig.suptitle("Vector calculus: grad, curl, div", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "vector_calculus.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
