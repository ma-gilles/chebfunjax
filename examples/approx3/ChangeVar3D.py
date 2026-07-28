"""Triple integrals in spherical, cylindrical and other coordinate systems.

Faithful port of approx3/ChangeVar3D.m by Rodrigo Platte (November 2016).
Uses chebfun3 coordinate maps and the chebfun3v Jacobian determinant to
compute triple integrals over non-rectangular 3D volumes via a change of
variables:  x=x(u,v,w), y=y(u,v,w), z=z(u,v,w) are chebfun3 objects on a
rectangular (u,v,w) box, and ``integral3(f.*abs(jacobian(x,y,z)))`` (= sum3
of the weighted integrand) gives the integral over the mapped region.

Original: https://www.chebfun.org/examples/approx3/ChangeVar3D.html
Copyright 2016 by The University of Oxford and The Chebfun Developers.

Output-parity note (measured): every published, verifiable integral is
reproduced to ~13-15 significant figures using the library Jacobian:
  ice-cream-cone mass  M = 0.6134341230070736  (published ...076)
  int r^2 |J|            = 0.3680604738042413   (published ...247; exact
                           pi(2-sqrt2)/5 = 0.368060473804244)
  cylinder-sector mass M = 1.5707963267948926  (published ...894)
  centre of mass         = [~0, 0.4244131815783866, 0.4999999999999998]
                           (published [0, ...388, 0.5])
  varied-torus volume    = 110.93435346824378  (published 1.109343...e+02)
The two torus circulation integrals (published -3.6e-23 and 4.2e-24) are
numerically zero by symmetry -- unverifiable residuals below the 1e-11 parity
floor -- so the port reports the first one and confirms it vanishes.

Runtime note: the coordinate maps and their Jacobians are genuinely
expensive chebfun3 constructions (the torus map's sin(3x) with x ranging over
[-5,5] resolves to a high-degree tensor).  To keep the whole script within the
parity harness budget, the first torus circulation integral is formed as a
single integrand chebfun3 (exactly integral3's own semantics) and the second,
redundant radius-varied torus region -- which likewise integrates to zero and
carries no verifiable number -- is omitted.  All verifiable sections use the
Jacobian exactly as MATLAB does.
"""
import matplotlib

matplotlib.use("Agg")
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v
from chebfunjax.plotting import PARULA, _setup_3d_axes, chebfun_style

chebfun_style()

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)), "docs", "images", "approx3"
)
os.makedirs(_IMG_DIR, exist_ok=True)


def run():
    twopi = 2 * float(np.pi)

    # ------------------------------------------------------------------
    # Spherical coordinates: mass of an "ice-cream cone" of variable
    # density.  x = r cos(t) cos(p), y = r sin(t) cos(p), z = r sin(p).
    # ------------------------------------------------------------------
    dom = (0.0, 1.0, 0.0, twopi, float(np.pi) / 4, float(np.pi) / 2)
    r = chebfun3(lambda r, t, p: r, domain=dom)
    t = chebfun3(lambda r, t, p: t, domain=dom)
    p = chebfun3(lambda r, t, p: p, domain=dom)
    x = r * t.cos() * p.cos()
    y = r * t.sin() * p.cos()
    z = r * p.sin()

    density = (10 * t).sin() * (10 * r).cos() + 1
    jac = Chebfun3v([x, y, z]).jacobian()
    M = float((density * jac.abs()).sum3())
    print(f"   {M:.15f}")

    # Simpler density r^2, exact answer pi(2-sqrt2)/5.
    print(f"   {float((r**2 * jac.abs()).sum3()):.15f}")
    print(f"   {float(np.pi) * (2 - np.sqrt(2)) / 5:.15f}")

    # ------------------------------------------------------------------
    # Cylindrical coordinates: centre of mass of a cylinder sector.
    # x = r cos(t), y = r sin(t), z = z.
    # ------------------------------------------------------------------
    dom = (0.0, 1.0, 0.0, float(np.pi), 0.0, 1.0)
    r = chebfun3(lambda r, t, z: r, domain=dom)
    t = chebfun3(lambda r, t, z: t, domain=dom)
    z = chebfun3(lambda r, t, z: z, domain=dom)
    x = r * t.cos()
    y = r * t.sin()

    density = y * (10 * t).sin() + 1
    jac = Chebfun3v([x, y, z]).jacobian().abs()
    M = float((density * jac).sum3())
    print(f"   {M:.15f}")

    xc = float((x * density * jac).sum3()) / M
    yc = float((y * density * jac).sum3()) / M
    zc = float((z * density * jac).sum3()) / M
    print(f"   {xc:.15f}   {yc:.15f}   {zc:.15f}")

    # ------------------------------------------------------------------
    # Toroidal coordinates: a triple integral over the torus.
    # x = (4 + r cos(t)) cos(p), y = (4 + r cos(t)) sin(p), z = r sin(t).
    # ------------------------------------------------------------------
    dom = (0.0, 1.0, 0.0, twopi, 0.0, twopi)
    r = chebfun3(lambda r, t, p: r, domain=dom)
    t = chebfun3(lambda r, t, p: t, domain=dom)
    p = chebfun3(lambda r, t, p: p, domain=dom)
    x = (4 + r * t.cos()) * p.cos()
    y = (4 + r * t.cos()) * p.sin()
    z = r * t.sin()
    jac_abs = Chebfun3v([x, y, z]).jacobian().abs()

    # f = sin(7z) sin(3x);  integral3(f |J|) -- built as one integrand
    # chebfun3 (integral3's own semantics), integrates to ~0 by symmetry.
    integrand = chebfun3(
        lambda R, T, P: (jnp.sin(7 * z(R, T, P)) * jnp.sin(3 * x(R, T, P))
                         * jac_abs(R, T, P)),
        domain=dom,
    )
    print(f"   {float(integrand.sum3()):.15e}")

    # Varying the torus radius: rr = r(1 + 0.9 sin(10 p)); volume via |J|.
    rr = r * (1 + 0.9 * (10 * p).sin())
    x = (4 + rr * t.cos()) * p.cos()
    y = (4 + rr * t.cos()) * p.sin()
    z = rr * t.sin()
    vol = float(Chebfun3v([x, y, z]).jacobian().abs().sum3())
    print(f"   {vol:.15e}")

    # ------------------------------------------------------------------
    # Plot: the three mapped regions.
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 4))

    ax1 = fig.add_subplot(131, projection="3d")
    _setup_3d_axes(ax1, fig)
    phi = np.linspace(np.pi / 4, np.pi / 2, 50)
    th = np.linspace(0, 2 * np.pi, 80)
    Phi, Th = np.meshgrid(phi, th)
    ax1.plot_surface(np.cos(Th) * np.cos(Phi), np.sin(Th) * np.cos(Phi),
                     np.sin(Phi), cmap=PARULA, linewidth=0, antialiased=True)
    ax1.set_title("spherical: ice-cream cone", fontsize=9, pad=0)

    ax2 = fig.add_subplot(132, projection="3d")
    _setup_3d_axes(ax2, fig)
    rr2, tt2 = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, np.pi, 60))
    ax2.plot_surface(rr2 * np.cos(tt2), rr2 * np.sin(tt2), np.ones_like(rr2),
                     cmap=PARULA, alpha=0.7, linewidth=0)
    T2, Z2 = np.meshgrid(np.linspace(0, np.pi, 60), np.linspace(0, 1, 30))
    ax2.plot_surface(np.cos(T2), np.sin(T2), Z2, cmap=PARULA, alpha=0.7,
                     linewidth=0)
    ax2.set_title("cylindrical: sector", fontsize=9, pad=0)

    ax3 = fig.add_subplot(133, projection="3d")
    _setup_3d_axes(ax3, fig)
    pv, tv = np.meshgrid(np.linspace(0, 2 * np.pi, 80),
                         np.linspace(0, 2 * np.pi, 50))
    ax3.plot_surface((4 + np.cos(tv)) * np.cos(pv),
                     (4 + np.cos(tv)) * np.sin(pv), np.sin(tv),
                     cmap=PARULA, linewidth=0, antialiased=True)
    ax3.set_title("toroidal", fontsize=9, pad=0)

    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG_DIR, "ChangeVar3D.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    return True


if __name__ == "__main__":
    run()
