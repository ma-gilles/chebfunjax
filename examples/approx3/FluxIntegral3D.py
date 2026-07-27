"""Flux integral of a 3D vector field over a parametric surface.

The flux integral2(F, S) integrates a chebfun3v vector field F over a
surface given by a chebfun2v parametrization S:
    integral2(F, S) = int int F(S(u,v)) . (S_u x S_v) du dv.
Faithful port of approx3/FluxIntegral3D.m by Olivier Sete, June 2016.

See https://www.chebfun.org/examples/approx3/FluxIntegral3D.html
Copyright 2016 by The University of Oxford and The Chebfun Developers.
"""

import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v
from chebfunjax.plotting import chebfun_style

chebfun_style()

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                        "docs", "images", "approx3")
os.makedirs(_IMG_DIR, exist_ok=True)


def run():
    # MATLAB: dom = [-5 5 -5 5 -1 1];
    #         F = chebfun3v(@(x,y,z) x+y, @(x,y,z) x.*z+y, @(x,y,z) z, dom);
    dom = (-5.0, 5.0, -5.0, 5.0, -1.0, 1.0)
    F = Chebfun3v.from_functions(
        lambda x, y, z: x + y,
        lambda x, y, z: x * z + y,
        lambda x, y, z: z,
        domain=dom,
    )

    # Example 1: rippled disk
    # S = chebfun2v(@(r,t) r.*cos(t), @(r,t) r.*sin(t), @(r,t) cos(5*r), [0 5 0 2*pi]);
    S1 = Chebfun2v.from_functions(
        lambda r, t: r * jnp.cos(t),
        lambda r, t: r * jnp.sin(t),
        lambda r, t: jnp.cos(5.0 * r),
        domain=(0.0, 5.0, 0.0, 2.0 * np.pi),
    )
    flux1 = float(F.integral2(S1))
    print(f"integral2(F, S) [rippled disk] = {flux1:.15e}")

    # Example 2: lower half of the unit sphere
    # S = chebfun2v(@(phi,theta) sin(theta).*cos(phi), ..., [0 2*pi pi/2 pi]);
    S2 = Chebfun2v.from_functions(
        lambda phi, theta: jnp.sin(theta) * jnp.cos(phi),
        lambda phi, theta: jnp.sin(theta) * jnp.sin(phi),
        lambda phi, theta: jnp.cos(theta),
        domain=(0.0, 2.0 * np.pi, np.pi / 2.0, np.pi),
    )
    flux2 = float(F.integral2(S2))
    print(f"integral2(F, S) [lower hemisphere] = {flux2:.15f}")
    print(f"-2*pi = {-2.0 * np.pi:.15f}")

    # Plot the two surfaces.
    fig = plt.figure(figsize=(10, 4))
    for i, (S, ttl, uv) in enumerate([
        (S1, 'Rippled disk', (np.linspace(0, 5, 60), np.linspace(0, 2 * np.pi, 60))),
        (S2, 'Lower hemisphere', (np.linspace(0, 2 * np.pi, 60), np.linspace(np.pi / 2, np.pi, 60))),
    ]):
        U, V = np.meshgrid(*uv)
        Xs = np.asarray(S.components[0](jnp.array(U), jnp.array(V)))
        Ys = np.asarray(S.components[1](jnp.array(U), jnp.array(V)))
        Zs = np.asarray(S.components[2](jnp.array(U), jnp.array(V)))
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        ax.plot_surface(Xs, Ys, Zs, cmap='viridis', alpha=0.9)
        ax.set_title(ttl, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG_DIR, 'FluxIntegral3D.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # The hemisphere flux equals -2*pi.
    assert abs(flux2 - (-2.0 * np.pi)) < 1e-8, "hemisphere flux should be -2*pi"

    print("FluxIntegral3D: done")
    return True


if __name__ == "__main__":
    run()
