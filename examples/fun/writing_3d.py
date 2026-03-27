"""Writing a message in 3D.

The scribble command produces a chebfun of piecewise linear complex values.
By treating the real part as x, imaginary part as y, and adding a z
component, we can write messages on 3D surfaces.
Translated from fun/Writing3D.m.

Original: https://www.chebfun.org/examples/fun/Writing3D.html
Author: Nick Trefethen, November 2010
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()



def letter_3d(ch, x_off=0.0, y_off=0.0, z_func=None):
    """Return 3D curve for a capital letter."""
    # Simple stroke definitions: list of 2D arrays, then map to 3D
    strokes_2d = {
        'C': [np.array([[0.4, 0.85], [0.15, 1.0], [0.0, 0.7],
                         [0.0, 0.3], [0.15, 0.0], [0.4, 0.15]])],
        'H': [np.linspace([[0,0],[0,1]], [[0,0],[0,1]], 10).reshape(-1,2),
              np.array([[0,0.5],[0.4,0.5]]),
              np.array([[0.4,0],[0.4,1]])],
        'E': [np.array([[0,0],[0,1]]),
              np.array([[0,1],[0.4,1]]),
              np.array([[0,0.5],[0.3,0.5]]),
              np.array([[0,0],[0.4,0]])],
        'B': [np.array([[0,0],[0,1]]),
              np.array([[0,1],[0.35,0.85],[0.4,0.65],[0.35,0.5],[0,0.5]]),
              np.array([[0,0.5],[0.4,0.35],[0.45,0.15],[0.4,0],[0,0]])],
        'Y': [np.array([[0,1],[0.2,0.5]]),
              np.array([[0.4,1],[0.2,0.5]]),
              np.array([[0.2,0.5],[0.2,0]])],
    }

    result = []
    for stroke in strokes_2d.get(ch, [np.array([[0,0],[0.4,0]])]):
        xs = stroke[:, 0] + x_off
        ys_2d = stroke[:, 1] + y_off
        if z_func is not None:
            zs = z_func(xs, ys_2d)
        else:
            zs = np.zeros_like(xs)
        result.append((xs, ys_2d, zs))
    return result


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/fun')
    os.makedirs(outdir, exist_ok=True)

    fig = plt.figure()

    # --- Panel 1: 2D piecewise-linear complex path (flat) ---
    ax1 = fig.add_subplot(131)

    chars = 'CHEBY'
    offsets_2d = np.arange(len(chars)) * 0.55
    for i, (ch, ox) in enumerate(zip(chars, offsets_2d)):
        strokes = letter_3d(ch, x_off=ox)
        for xs, ys, zs in strokes:
            z = xs + 1j * ys
            ax1.plot(np.real(z), np.imag(z), 'b-', linewidth=2.5)

    ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3)
    ax1.set_title('"CHEBY" as piecewise-linear\ncomplex path', fontsize=10)

    # --- Panel 2: On a wavy surface z = sin(pi*x)*sin(pi*y)/4 ---
    ax2 = fig.add_subplot(132, projection='3d')

    def z_wavy(x, y):
        return 0.2 * np.sin(np.pi * x * 3) * np.cos(np.pi * y * 2)

    # Background surface
    xs_bg = np.linspace(0, 2.5, 40)
    ys_bg = np.linspace(0, 1.2, 20)
    Xs, Ys = np.meshgrid(xs_bg, ys_bg)
    Zs = z_wavy(Xs, Ys)
    ax2.plot_surface(Xs, Ys, Zs, alpha=0.3, color='lightblue', linewidth=0)

    for i, (ch, ox) in enumerate(zip(chars, offsets_2d)):
        strokes = letter_3d(ch, x_off=ox, z_func=z_wavy)
        for xs, ys, zs in strokes:
            ax2.plot(xs, ys, zs + 0.05, 'r-', linewidth=2.5)

    ax2.set_title('"CHEBY" on wavy\nsurface', fontsize=10)
    ax2.set_box_aspect([1,0.5,0.3])
    ax2.set_axis_off()

    # --- Panel 3: On a sphere ---
    ax3 = fig.add_subplot(133, projection='3d')

    # Draw sphere
    u = np.linspace(0, 2*np.pi, 40)
    v2 = np.linspace(0, np.pi, 20)
    Xu = np.outer(np.cos(u), np.sin(v2))
    Yu = np.outer(np.sin(u), np.sin(v2))
    Zu = np.outer(np.ones_like(u), np.cos(v2))
    ax3.plot_surface(Xu, Yu, Zu, alpha=0.15, color='lightgray', linewidth=0)

    # Write on equatorial band by projecting (x, y) onto sphere surface
    # Map x in [0, 2.5] -> theta in [0, 2*pi], y in [0, 1] -> phi in [pi/2-0.4, pi/2+0.4]
    def to_sphere(x, y):
        theta = x / 2.7 * 2 * np.pi
        phi = np.pi / 2 + (y - 0.5) * 0.8
        xs3 = np.sin(phi) * np.cos(theta) * 1.01
        ys3 = np.sin(phi) * np.sin(theta) * 1.01
        zs3 = np.cos(phi) * 1.01
        return xs3, ys3, zs3

    for i, (ch, ox) in enumerate(zip(chars, offsets_2d)):
        strokes = letter_3d(ch, x_off=ox)
        for xs_2d, ys_2d, _ in strokes:
            xs3, ys3, zs3 = to_sphere(xs_2d, ys_2d)
            ax3.plot(xs3, ys3, zs3, 'b-', linewidth=2.5)

    ax3.set_title('"CHEBY" written\non a sphere', fontsize=10)
    ax3.set_box_aspect([1,1,1])
    ax3.set_axis_off()

    print("Writing3D: text as 3D curves")
    print("  2D: complex piecewise-linear path")
    print("  3D wavy: projected onto z = 0.2*sin(3πx)*cos(2πy)")
    print("  3D sphere: stereographically mapped to unit sphere")

    fig.suptitle('Writing a Message in 3D', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'writing_3d.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("writing_3d: done")
    return True


if __name__ == "__main__":
    run()
