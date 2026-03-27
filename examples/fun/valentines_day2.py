"""Happy Valentine's Day! (again).

A second valentine's day example featuring heart-shaped parametric
surfaces and romantic mathematical curves in 3D.
Translated from fun/ValentinesDay2.m.

Original: https://www.chebfun.org/examples/fun/ValentinesDay2.html
Author: Anonymous, February 2013
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



def heart_surface(u, v):
    """Parametric 3D heart surface."""
    # 3D heart parametrization
    x = np.sin(u)**3 * np.cos(v)
    y = np.cos(u)**3 * np.cos(v)
    z = np.sin(v)
    return x, y, z


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/fun')
    os.makedirs(outdir, exist_ok=True)

    fig = plt.figure(figsize=(15, 5))

    # --- Panel 1: 3D heart surface ---
    ax1 = fig.add_subplot(131, projection='3d')
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(-np.pi/2, np.pi/2, 30)
    U, V = np.meshgrid(u, v)
    X = 2 * np.sin(U)**3
    Y = (13*np.cos(U) - 5*np.cos(2*U) - 2*np.cos(3*U) - np.cos(4*U)) / 8
    Z = V * 0.5  # give height to 2D heart

    ax1.plot_surface(X, Y, Z, color='crimson', alpha=0.8)
    ax1.set_title('3D Heart surface', fontsize=10)
    ax1.set_axis_off()

    # --- Panel 2: Animated heart (multiple frames) ---
    ax2 = fig.add_subplot(132)
    t = np.linspace(0, 2*np.pi, 500)
    beats = [1.0, 1.1, 1.2, 1.1, 1.0, 0.9]
    colors2 = plt.cm.Reds(np.linspace(0.4, 1.0, len(beats)))
    for scale, col in zip(beats, colors2):
        x1 = scale * 16 * np.sin(t)**3
        y1 = scale * (13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t))
        ax2.fill(x1, y1, color=col, alpha=0.3, zorder=1)
    # Overlay final heart
    x1 = 16 * np.sin(t)**3
    y1 = 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)
    ax2.fill(x1, y1, color='red', alpha=0.8, zorder=2)
    ax2.plot(x1, y1, 'darkred', linewidth=1.5, zorder=3)
    ax2.set_aspect('equal'); ax2.axis('off')
    ax2.set_title('Beating heart (overlaid\nscaled versions)', fontsize=10)

    # --- Panel 3: Rose of hearts ---
    ax3 = fig.add_subplot(133)
    n_hearts = 6
    for i in range(n_hearts):
        angle = 2*np.pi * i / n_hearts
        cx = 20 * np.cos(angle)
        cy = 20 * np.sin(angle)
        scale = 0.35
        xh = scale * 16 * np.sin(t)**3 + cx
        yh = scale * (13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)) + cy
        color_h = plt.cm.hsv(i / n_hearts)
        ax3.fill(xh, yh, color=color_h, alpha=0.7, zorder=2)
        ax3.plot(xh, yh, color='darkred', linewidth=0.5, zorder=3)

    # Center heart
    xh0 = 16 * np.sin(t)**3
    yh0 = 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)
    ax3.fill(xh0, yh0, color='red', alpha=0.9, zorder=4)

    ax3.set_aspect('equal'); ax3.axis('off')
    ax3.set_title('Rose of hearts', fontsize=10)

    # Areas
    area_classic = abs(np.sum(x1[:-1] * np.diff(y1)))
    print(f"Valentine's Day 2:")
    print(f"  Classic heart area: {area_classic:.2f}")
    print(f"  Exact area: {180*np.pi:.2f}")
    print(f"  6 hearts in rose of radius 20")

    fig.suptitle("Happy Valentine's Day! (Chebfun2 Edition)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'valentines_day2.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("valentines_day2: done")
    return True


if __name__ == "__main__":
    run()
