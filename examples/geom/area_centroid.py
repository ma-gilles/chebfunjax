"""Area and centroid of a 2D region.

Areas of regions bounded by parametric curves are computed with Green's
theorem via Chebfun: A = sum(x.*diff(y)) = oint x dy.  Faithful port of
geom/Area.m.

Original: https://www.chebfun.org/examples/geom/Area.html
Author: Stefan Guettel, October 2010
"""

import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/geom')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2)

    # --- Epicycloid (m = 7) ----------------------------------------------
    # MATLAB: t = chebfun('t',[0,2*pi]); b=1; m=7; a=(m-1)*b;
    #   x = (a+b)*cos(t) - b*cos((a+b)/b*t);
    #   y = (a+b)*sin(t) - b*sin((a+b)/b*t);
    #   A = sum(x.*diff(y));  exact = pi*b^2*(m^2+m)
    t = cj.chebfun(lambda t: t, domain=(0.0, 2.0 * np.pi))
    b, m = 1.0, 7.0
    a = (m - 1.0) * b
    x = (a + b) * t.cos() - b * (((a + b) / b) * t).cos()
    y = (a + b) * t.sin() - b * (((a + b) / b) * t).sin()
    A = float((x * y.diff()).sum())
    exact = np.pi * b**2 * (m**2 + m)
    print(f"A = {A:.15e}")
    print(f"exact = {exact:.15e}")

    tt = np.linspace(0.0, 2.0 * np.pi, 2000)
    axes[0].fill(np.asarray(x(jnp.array(tt))), np.asarray(y(jnp.array(tt))),
                 color=[0.6, 0.6, 1.0], alpha=0.7)
    axes[0].set_aspect('equal')
    axes[0].set_title(f'Epicycloid (m={int(m)}), A = {A:.4f}', fontsize=10)

    # --- Perturbed unit circle: area (= pi) and centroid -----------------
    # MATLAB: z = exp(1i*t) + (1+1i)*sin(6*t).^2;
    #   A = sum(real(z).*diff(imag(z)));  [A; pi]
    #   c = sum(diff(z).*z.*conj(z))/(2i*A);
    z = cj.chebfun(lambda t: jnp.exp(1j * t) + (1 + 1j) * jnp.sin(6 * t)**2,
                   domain=(0.0, 2.0 * np.pi))
    A2 = float((z.real() * z.imag().diff()).sum())
    print(f"[A; pi] = {A2:.15f}   {np.pi:.15f}")
    c = complex((z.diff() * z * z.conj()).sum()) / (2j * A2)

    zv = np.asarray(z(jnp.array(tt)))
    axes[1].fill(np.real(zv), np.imag(zv), color=[0.6, 1.0, 0.6], alpha=0.7)
    axes[1].plot(c.real, c.imag, color='#D95319', marker='+', linestyle='none',
                 markersize=15, markeredgewidth=2, label='centroid')
    axes[1].set_aspect('equal')
    axes[1].legend(fontsize=9)
    axes[1].set_title(f'Perturbed circle, A = {A2:.4f} (= pi)', fontsize=10)

    fig.suptitle("Area via Green's theorem", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'area_centroid.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    print("area_centroid: done")
    return True


if __name__ == "__main__":
    run()
