"""Eigenvalue landscapes.

Translation of linalg/EigLandscapes.m by Nick Trefethen
(November 2019): the smallest eigenvalues of a two-parameter Hermitian
family B + xC + yD as chebfun2 surfaces — analytic (generic complex
Hermitian) vs kinked where eigenvalues cross (real symmetric,
requiring fixed-grid construction).

randn draws are not bit-reproducible between MATLAB and numpy; the
landscapes are our own draws of the same families.

Original: https://www.chebfun.org/examples/linalg/EigLandscapes.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _public_savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')

_CAPTURE = os.environ.get("EIGLANDSCAPES_CAPTURE_DIR")


def _savefig(fig, path, **kwargs):
    if _CAPTURE:
        import pickle

        os.makedirs(_CAPTURE, exist_ok=True)
        with open(os.path.join(_CAPTURE, os.path.basename(path) + ".pickle"), "wb") as output:
            pickle.dump(fig, output)
    from mpl_toolkits.mplot3d import Axes3D

    original = Axes3D.apply_aspect
    try:
        if any(isinstance(ax, Axes3D) for ax in fig.axes):
            # MATLAB renders a rectangular normalized axes panel. The pinned
            # Matplotlib 3D default shrinks that panel to a square. Retain the
            # panel rectangle during this sequential figure save only.
            Axes3D.apply_aspect = _rectangular_3d_aspect
        _public_savefig(fig, path, **kwargs)
    finally:
        Axes3D.apply_aspect = original


def _rectangular_3d_aspect(self, position=None):
    position = self.get_position(original=True) if position is None else position
    self._set_position(position, "active")


def _configure_surface_axes(ax, view=None):
    from matplotlib.ticker import MaxNLocator

    ax.set_proj_type("ortho")
    ax.set_box_aspect((1, 1, .8), zoom=1)
    # MATLAB view azimuth is measured from -y; Matplotlib from +x.
    azimuth, elevation = view if view else (-37.5, 30)
    ax.view_init(elev=elevation, azim=azimuth-90)
    ax.set_xticks([0, .1, .2, .3, .4] if view else [-1, 0, 1])
    ax.set_yticks([0, .2, .4] if view else [-1, 0, 1])
    if view:
        ax.set_zticks([-5.5, -5, -4.5, -4])
    else:
        ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(labelsize=8, pad=1)
    ax.zaxis.set_tick_params(pad=6)
    ax.set_xlabel("x", fontsize=9, labelpad=8)
    ax.set_ylabel("y", fontsize=9, labelpad=3)
    ax.figure.subplots_adjust(left=.10, right=.96, bottom=.20, top=.88)


def _capture_grid(name, X, Y, Z):
    if _CAPTURE:
        os.makedirs(_CAPTURE, exist_ok=True)
        np.savez(os.path.join(_CAPTURE, name), X=X, Y=Y, Z=Z)


N = 8
FIG = [0]


def _eig_op(B, C, D, k):
    """Ascending Hermitian eigenvalues in bounded 256-matrix JAX batches.

    Preserve A=B+x*C+y*D and the full requested coordinate grid. Padding
    repeats only the last coordinate; padded outputs are discarded.
    """
    B, C, D = jnp.asarray(B), jnp.asarray(C), jnp.asarray(D)
    batch_size = 256

    @jax.jit
    def evaluate(matrices):
        return jnp.linalg.eigvalsh(matrices, symmetrize_input=False)[:, k]

    def op(X, Y):
        X, Y = jnp.broadcast_arrays(jnp.atleast_1d(jnp.asarray(X, dtype=jnp.float64)),
                                     jnp.atleast_1d(jnp.asarray(Y, dtype=jnp.float64)))
        shape = X.shape
        x, y = X.reshape(-1), Y.reshape(-1)
        if x.size == 0:
            return jnp.empty(shape, dtype=jnp.float64)
        parts = []
        for start in range(0, x.size, batch_size):
            count = min(batch_size, x.size-start)
            xb, yb = x[start:start+count], y[start:start+count]
            if count < batch_size:
                xb = jnp.pad(xb, (0, batch_size-count), mode="edge")
                yb = jnp.pad(yb, (0, batch_size-count), mode="edge")
            matrices = B + xb[:, None, None] * C + yb[:, None, None] * D
            parts.append(evaluate(matrices)[:count])
        return jnp.concatenate(parts).reshape(shape)
    return op


def _surf(fs, fname, view=None, axis3d=None):
    FIG[0] += 1
    fig = plt.figure(figsize=(6, 2.53), dpi=100)
    ax = fig.add_subplot(projection="3d")
    xs = np.linspace(axis3d[0], axis3d[1], 80) if axis3d else np.linspace(-1, 1, 80)
    ys = np.linspace(axis3d[2], axis3d[3], 80) if axis3d else xs
    X, Y = np.meshgrid(xs, ys)
    for component, f in enumerate(fs):
        Z = np.asarray(f(jnp.asarray(X), jnp.asarray(Y)))
        _capture_grid(f"EigLandscapes_{FIG[0]:02d}_surface_{component}.npz", X, Y, Z)
        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9,
                        axlim_clip=bool(axis3d))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if axis3d:
        ax.set_xlim(axis3d[0], axis3d[1])
        ax.set_ylim(axis3d[2], axis3d[3])
        ax.set_zlim(axis3d[4], axis3d[5])
    ax.set_title(fname, fontsize=10)
    _configure_surface_axes(ax, view)
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(
        _IMG, f"EigLandscapes_{FIG[0]:02d}.png"), size=(600, 253))
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    rs = np.random.RandomState(1)

    def herm():
        M = rs.randn(N, N) + 1j * rs.randn(N, N)
        return M + M.conj().T

    B, C, D = herm(), herm(), herm()
    f1 = cj.chebfun2(_eig_op(B, C, D, 0))
    f2 = cj.chebfun2(_eig_op(B, C, D, 1))
    _surf([f1, f2], "first two eigenvalues")

    gap = f2 - f1
    mingap, _loc = gap.min2()
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(6, 2.53), dpi=100)
    xs = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(xs, xs)
    Z = np.asarray(gap(jnp.asarray(X), jnp.asarray(Y)))
    _capture_grid(f"EigLandscapes_{FIG[0]:02d}_contour.npz", X, Y, Z)
    cs = ax.contour(X, Y, Z, levels=np.arange(0, 3.01, 0.25))
    fig.colorbar(cs, ax=ax)
    cs.set_clim(0, 3)
    ax.set_aspect("equal")
    ax.set_title(f"min gap = {float(mingap):.5g}", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, f"EigLandscapes_{FIG[0]:02d}.png"), size=(600, 253))
    plt.close(fig)

    # real symmetric: eigenvalues can cross, surfaces have kinks;
    # fixed 512-point grids as in the MATLAB example
    rs2 = np.random.RandomState(8)

    def sym():
        M = rs2.randn(N, N)
        return M + M.T

    B, C, D = sym(), sym(), sym()
    npts = 512
    f1 = cj.chebfun2(_eig_op(B, C, D, 0), n=npts)
    f2 = cj.chebfun2(_eig_op(B, C, D, 1), n=npts)
    _surf([f1, f2], "first two eigenvalues")
    # gap = f2-f1; mingap = min2(gap); contour(gap,levels), axis equal
    gap = f2 - f1
    mingap, _loc = gap.min2()
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(6, 2.53), dpi=100)
    xs = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(xs, xs)
    Z = np.asarray(gap(jnp.asarray(X), jnp.asarray(Y)))
    _capture_grid(f"EigLandscapes_{FIG[0]:02d}_contour.npz", X, Y, Z)
    cs = ax.contour(X, Y, Z, levels=np.arange(0, 3.01, 0.25))
    fig.colorbar(cs, ax=ax)
    cs.set_clim(0, 3)
    ax.set_aspect("equal")
    ax.set_title(f"min gap = {float(mingap):.5g}", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, f"EigLandscapes_{FIG[0]:02d}.png"), size=(600, 253))
    plt.close(fig)
    _surf([f1, f2], "",
          view=(-27, 18), axis3d=(-.1, .4, -.1, .4, -5.5, -4))


if __name__ == "__main__":
    run()
