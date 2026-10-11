"""Laplace equation on the unit ball with captured random boundary data.

Translation of the native sphere/LaplaceBall.m source example. The fixed
MATLAB R2025b input fixture supplies only the 1,024 normal draws at the
public randnfunsphere np.random.randn boundary. The public generator still
constructs the Spherefun; no coefficients or computed outputs are substituted.
The captured input vector reproduces one native sample only; general RNG,
numerical-output, and rendering parity require separate evidence.

Original: https://www.chebfun.org/examples/sphere/LaplaceBall.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import contextlib
import json
import pickle
import time

import matplotlib

matplotlib.use("Agg")
import os
import sys
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from chebfunjax._colormaps import parula_colormap
from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.chebfun1d.randfuns import randnfunsphere
from chebfunjax.plotting import chebfun_style, plot_ball_slices
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = Path(__file__).resolve().parent
_FIXTURE = Path(os.environ.get("LAPLACEBALL_INPUT_FIXTURE", _HERE / "laplaceball_inputs_r2025b.npz"))
_IMAGE_DIR = Path(os.environ.get("LAPLACEBALL_IMAGE_DIR", _HERE / "images"))
_CAPTURE_DIR = Path(os.environ.get("LAPLACEBALL_CAPTURE_DIR", _HERE / "captures"))
FIG = [0]
LAMBDA = 0.2


@contextlib.contextmanager
def _native_draws_at_randn(draws):
    """Inject captured normal inputs while leaving public construction intact."""
    original = np.random.randn
    used = {"count": 0}

    def take_draws(*shape):
        if used["count"] != 0:
            raise RuntimeError("randnfunsphere requested normal inputs more than once")
        if len(shape) != 1 or int(shape[0]) != draws.size:
            raise RuntimeError(f"unexpected randn shape: {shape!r}")
        used["count"] += 1
        return draws.copy()

    np.random.randn = take_draws
    try:
        yield used
    finally:
        np.random.randn = original


def _boundary_data():
    with np.load(_FIXTURE, allow_pickle=False) as data:
        draws = np.asarray(data["normal_draws"], dtype=np.float64).copy()
    if draws.shape != ((int(np.floor(2 * np.pi / LAMBDA)) + 1) ** 2,):
        raise ValueError(f"wrong frozen input shape: {draws.shape}")
    with _native_draws_at_randn(draws) as used:
        h = randnfunsphere(LAMBDA)
    if used["count"] != 1:
        raise RuntimeError("randnfunsphere did not consume the one frozen input vector")
    return h


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    _savefig(fig, str(_IMAGE_DIR / f"LaplaceBall_{FIG[0]:02d}.png"), size=(600, 253))
    with (_IMAGE_DIR / f"LaplaceBall_{FIG[0]:02d}.pickle").open("wb") as stream:
        pickle.dump(fig, stream, protocol=5)
    plt.close(fig)


def _capture(name, value):
    """Retain actual completed objects for rendering without another solve."""
    _CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    with (_CAPTURE_DIR / f"{name}.pickle").open("wb") as stream:
        pickle.dump(value, stream, protocol=5)
    leaves, structure = jax.tree_util.tree_flatten(value)
    arrays = {f"leaf_{i:04d}": np.asarray(leaf) for i, leaf in enumerate(leaves)}
    np.savez(_CAPTURE_DIR / f"{name}_leaves.npz", **arrays)
    (_CAPTURE_DIR / f"{name}_metadata.json").write_text(
        json.dumps({"type": type(value).__name__, "tree_structure": str(structure),
                    "leaves": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                               for k, v in arrays.items()}}, indent=2) + "\n")
    (_CAPTURE_DIR / "last_completed_stage.json").write_text(
        json.dumps({"stage": name, "unix_seconds": time.time()}) + "\n")


def _sphere_layout(ax, cb):
    """Place source sphere view and colorbar inside the final canvas."""
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    ax.view_init(elev=30, azim=-127.5)
    ax.set_proj_type("ortho")
    ax.tick_params(pad=-2)
    ax.tick_params(axis="x", pad=-8)
    ax.set_box_aspect([1, 1, 1], zoom=1.065)
    ax.set_position([-0.027, 0.094, 0.64, 0.815])
    cb.ax.set_box_aspect(None)
    cb.ax.set_aspect("auto")
    cb.ax.set_position([0.84, 0.11, 0.032, 0.815])
    cb.ax.tick_params(labelsize=8)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))


def _render(h, u, uinner):
    """Render the three source figures from actual retained functions."""
    _IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    # Cached page uses the exact R2017a-era parula table with 64 entries.
    palette = parula_colormap(64)
    fig = plt.figure(figsize=(6.0, 2.53))
    ax = fig.add_subplot(111, projection="3d")
    fig, ax, mappable = h.plot(ax=ax, clim=(-2, 2), cmap=palette, return_mappable=True)
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    _sphere_layout(ax, cb)
    ax.set_axis_off()
    _save(fig)
    fig = plt.figure(figsize=(6.0, 2.53))
    ax = fig.add_subplot(111, projection="3d")
    plot_ball_slices(u, ax=ax, azim=-127.5, cmap=palette)
    _save(fig)
    fig = plt.figure(figsize=(6.0, 2.53))
    ax = fig.add_subplot(111, projection="3d")
    fig, ax, mappable = uinner.plot(ax=ax, cmap=palette, return_mappable=True)
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    _sphere_layout(ax, cb)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    _save(fig)


def render_saved():
    """Replay plotting only from captures under the same pinned source."""
    values = []
    for name in ("boundary", "solution", "inner_sphere"):
        with (_CAPTURE_DIR / f"{name}.pickle").open("rb") as stream:
            values.append(pickle.load(stream))
    _render(*values)


def run():
    _IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    h = _boundary_data()

    _capture("boundary", h)

    # The sphere's angular point (lambda=0, theta=pi/2) is (1,0,0).
    h_xaxis = float(h(0.0, np.pi / 2))
    print("ans =")
    print(f"  {h_xaxis:.15f}")
    print("ans =")
    print(f"  {float(h(0.0, np.pi / 2)):.15f}")
    meanh = float(h.mean2())
    print("meanh =")
    print(f"  {meanh:.15f}")

    # Source cells: [a,b]=length(h); m=ceil(max(a,b)); u=poisson(zero,h,m).
    a, b = h.length()
    m = int(np.ceil(max(a, b)))
    zero = Ballfun.from_function(lambda x, y, z: 0.0 * x)
    _capture("rhs", zero)
    u = Ballfun.poisson(zero, h, m)
    _capture("solution", u)

    u_xaxis = float(u.feval(1.0, 0.0, 0.0, coord="cartesian"))
    print("ans =")                         # h(1,0,0), via sphere angles
    print(f"  {h_xaxis:.15f}")
    print("ans =")                         # u(1,0,0)
    print(f"  {u_xaxis:.15f}")

    long = -1.26 * np.pi / 180
    lat = 51.75 * np.pi / 180
    h_oxford = float(h(long, np.pi / 2 - lat))
    u_oxford = float(u.feval(1.0, long, np.pi / 2 - lat, coord="spherical"))
    print("ans =")
    print(f"  {h_oxford:.15f}")
    print("ans =")
    print(f"  {u_oxford:.15f}")

    u_origin = float(u.feval(0.0, 0.0, 0.0, coord="cartesian"))
    print("meanh =")
    print(f"  {meanh:.15f}")
    print("ans =")
    print(f"  {u_origin:.15f}")

    uinner = u.to_spherefun(0.5)
    _capture("inner_sphere", uinner)
    mean_inner = float(uinner.mean2())
    print("meanh =")
    print(f"  {meanh:.15f}")
    print("ans =")
    print(f"  {mean_inner:.15f}")
    np.savez(_CAPTURE_DIR / "printed_values.npz", h_xaxis=h_xaxis, meanh=meanh,
             u_xaxis=u_xaxis, h_oxford=h_oxford, u_oxford=u_oxford,
             u_origin=u_origin, mean_inner=mean_inner, m=m)
    _render(h, u, uinner)


if __name__ == "__main__":
    if sys.argv[1:] == ["--render-saved"]:
        render_saved()
    elif not sys.argv[1:]:
        run()
    else:
        raise SystemExit("expected no arguments or --render-saved")
