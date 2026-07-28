"""Integration of scalar functions over 3D curves.

Faithful port of approx3/LineIntegral3D.m by Behnam Hashemi (June 2016).
Computes the line integral int_C f ds of a Chebfun3 scalar field along a 3D
parametric curve using the chebfun3 ``integral`` operator (arc-length
weighting), for a sine-wave curve on the unit sphere and a spherical helix.

Original: https://www.chebfun.org/examples/approx3/LineIntegral3D.html
Copyright 2016 by The University of Oxford and The Chebfun Developers.

Output-parity note (measured): both published line integrals reproduce to
~14-15 significant figures via the library integral (10.746250564473513 and
-0.040586850422702).  The prior port used a hand-rolled trapezoid (~1e-6);
this uses ``Chebfun3.integral(curve, domain)`` directly.
"""
import matplotlib

matplotlib.use("Agg")
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.plotting import chebfun_style

chebfun_style()

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)), "docs", "images", "approx3"
)
os.makedirs(_IMG_DIR, exist_ok=True)


def run():
    # ------------------------------------------------------------------
    # Example 1: cos(x + y z) over a sine-wave curve on the unit sphere.
    #   C(t) = (cos t . s, sin t . s, r cos(pt)),  s = sqrt(q^2 - r^2 cos^2(pt))
    # ------------------------------------------------------------------
    p, q, r = 10, 1, 0.3

    def C1(t):
        s = jnp.sqrt(q**2 - r**2 * jnp.cos(p * t)**2)
        return jnp.stack([jnp.cos(t) * s, jnp.sin(t) * s, r * jnp.cos(p * t)],
                         axis=-1)

    f1 = chebfun3(lambda x, y, z: jnp.cos(x + y * z))
    I = float(f1.integral(curve=C1, domain=(0, 2 * float(np.pi))))
    print("I =")
    print(f"  {I:.15f}")

    # ------------------------------------------------------------------
    # Example 2: x + y z over a spherical helix (loxodrome).
    #   C(t) = (sin(t/2r) cos t, sin(t/2r) sin t, cos(t/2r)),  r = 5
    # ------------------------------------------------------------------
    r2 = 5

    def C2(t):
        return jnp.stack([jnp.sin(t / (2 * r2)) * jnp.cos(t),
                          jnp.sin(t / (2 * r2)) * jnp.sin(t),
                          jnp.cos(t / (2 * r2))], axis=-1)

    f2 = chebfun3(lambda x, y, z: x + y * z)
    I = float(f2.integral(curve=C2, domain=(0, 10 * float(np.pi))))
    print("I =")
    print(f"  {I:.15f}")

    # ------------------------------------------------------------------
    # Plot: the two curves on the unit sphere.
    # ------------------------------------------------------------------
    from chebfunjax.plotting import _setup_3d_axes

    fig = plt.figure(figsize=(10, 4.5))
    for idx, (C, ttl, tmax) in enumerate([
            (C1, "sine-wave curve", 2 * np.pi),
            (C2, "spherical helix", 10 * np.pi)]):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        _setup_3d_axes(ax, fig)
        tt = np.linspace(0, tmax, 4000)
        xyz = np.asarray(C(jnp.asarray(tt)))
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], lw=1.0)
        ax.set_title(ttl, fontsize=10)

    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG_DIR, "LineIntegral3D.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    return True


if __name__ == "__main__":
    run()
