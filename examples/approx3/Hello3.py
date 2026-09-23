"""Hello 3D World — Chebfun3 from discrete data.

Translation of approx3/Hello3.m by Olivier Sète (June 2016): a binary
40x40x40 tensor spelling "HELLO" becomes a chebfun3 through the 'equi'
constructor, and its isosurfaces at 0.5 and -0.1 show the letters and
the interpolant's Gibbs ripples.

Original: https://www.chebfun.org/examples/approx3/Hello3.html
Copyright 2016 by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.plotting import chebfun_style, isosurface_chebfun3
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx3')

# MATLAB view([-2.5, -1, 0.4]) as matplotlib (elev, azim) in degrees.
_ELEV = float(np.degrees(np.arctan2(0.4, np.hypot(-2.5, -1))))
_AZIM = float(np.degrees(np.arctan2(-2.5, 1))) - 90


def _iso(f, level, k):
    fig = plt.figure(figsize=(6.0, 2.7))
    # A square 3-D box as wide as the 600x270 canvas (axis equal/off).
    ax = fig.add_axes([0, -0.6, 1, 2.2], projection="3d")
    isosurface_chebfun3(f, level, ax=ax, n_pts=81, alpha=1.0)
    # camlight: re-add the triangulation with Gouraud-like face shading
    # lit from the camera direction; axis equal hugs the surface.
    mesh = ax.collections[-1]
    tri = np.asarray(mesh._vec[:3]).T.reshape(-1, 3, 3)
    mesh.remove()
    ax.add_collection3d(Poly3DCollection(
        tri, shade=True, facecolors="#2BB7A0", linewidths=0,
        lightsource=LightSource(azdeg=_AZIM + 60, altdeg=_ELEV + 30)))
    lo, hi = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(hi - lo), zoom=0.8)
    ax.view_init(elev=_ELEV, azim=_AZIM)
    ax.set_axis_off()
    _savefig(fig, os.path.join(_IMG, f"Hello3_{k:02d}.png"))
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    # The letters (MATLAB 1-based ranges a:b become a-1:b).
    A = np.zeros((15, 40))
    A[1:9, 1:3] = 1; A[4:6, 3:5] = 1; A[1:9, 5:7] = 1; A[2:10, 9:11] = 1
    A[2:4, 9:15] = 1; A[5:7, 9:15] = 1; A[8:10, 9:15] = 1; A[3:11, 17:19] = 1
    A[9:11, 17:24] = 1; A[4:12, 25:27] = 1; A[10:12, 25:31] = 1
    A[5:13, 33:35] = 1; A[5:13, 37:39] = 1; A[5:7, 35:37] = 1
    A[11:13, 35:37] = 1

    A = np.vstack([np.zeros((14, 40)), A, np.zeros((11, 40))])
    A = np.fliplr(np.flipud(A))
    B = np.zeros((40, 40, 40))
    for k in range(17, 21):
        B[k, :, :] = A

    f = Chebfun3.from_equidata(B)          # chebfun3(B, 'equi')

    f = f.permute([1, 3, 2])
    _iso(f, 0.5, 1)

    _iso(f, -0.1, 2)


if __name__ == "__main__":
    run()
