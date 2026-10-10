"""Computing with an atmospheric dataset in Spherefun.

Source: sphere/AtmosphericTemperature.m, Chebfun examples f4b9ea46.
With no arguments, use the project .atmospheric_data.mat cache, downloading
the pinned input when missing. --data and ATMOSPHERIC_DATA override that path.
All printed values and ten figures are computed, without reference output.

Original: https://www.chebfun.org/examples/sphere/AtmosphericTemperature.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# uses-numpy: pinned MAT input and host-side plotting options only.
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.io import loadmat

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from chebfunjax.plotting import (  # noqa: E402
    chebfun_style,
    matlab_explicit_camera,
    matlab_plot,
    matlab_view,
    plot_earth,
    save_chebfun_figure,
)
from chebfunjax.spherefun.spherefun import Spherefun  # noqa: E402

_MAT_URL = (
    "https://raw.githubusercontent.com/chebfun/examples/"
    "f4b9ea46cfc2f52f20a844627f4a74d0bb10098c/sphere/AtmosphericData.mat"
)
_DATA_SHA256 = "27549e83460110925ee85c8f7ae6875e7f8ed749f30eaebd5f6290c45f3108bd"


def _print_source_spherefun_display(fun):
    """Native display fields, calculated from the current representation."""
    print("\nf =\n")
    print("   spherefun object")
    print("       domain        rank    vertical scale")
    print(f"     unit sphere  {int(fun.rank):6d}          {float(fun.vscale()):3.2g}\n")


# Explicit R2025b/7574c77 captured camera properties (not an automatic policy).
# Metadata SHA3754684ed299c6bfb597114ef5e22a3739ebcf00da35a31dbcf35866ac96c8bc.
_SOURCE_SURFACE_CAMERA = {'position': (13.268278963378766, -11.133407984528386, 0), 'target': (0, 0, 0), 'up': (0, 0, 1), 'view_angle': 6.608610360311924, 'data_aspect': (1, 1, 1)}
_SOURCE_CONTOUR_CAMERA = {'position': (13.217789156120169, -11.091042005879483, 1.5095817461034662), 'target': (0, 0, 0), 'up': (0, 0, 1), 'view_angle': 7.392854781564241, 'data_aspect': (1, 1, 1)}

def _print_source_ans(value):
    """Computed scalar output; MATLAB session formatting is not inferred."""
    print("\nans =\n")
    print(f"{float(value):20.15f}\n")


def _website_jet():
    """Discrete jet64 for the 2016 website's pre-R2019b graphics default.

    MathWorks jet version history documents 64 before R2019b. These clipped
    linear RGB ramps independently express jet64, not Matplotlib's jet LUT.
    No proprietary sampled palette table is redistributed.
    """
    levels = np.arange(1, 65, dtype=float) / 64
    channels = np.array([3., 2., 1.])
    colors = np.clip(1.5 - np.abs(4 * levels[:, None] - channels), 0., 1.)
    return ListedColormap(colors, name="website_jet64")


def _website_colorbar_layout(fig, ax, mappable):
    """Measured figure01 presentation; not an inferred HG2 layout engine.

    Reference 600x270 PNG bar is x477..515, y20..239. The sphere's saturated
    footprint is x128..347, y20..239. The 218-pixel viewport includes an
    explicit two-pixel allowance for Agg Gouraud edge rasterization.
    The unit-sphere artist must retain that edge beyond the axes clip box.
    Field data, color limits and captured source camera remain independent.
    """
    fig.set_layout_engine(None)
    ax.set_position((78/600, 31/270, 320/600, 218/270))
    for artist in ax.collections:
        if getattr(artist, "_chebfun_sphere_radius", None) == 1.0:
            artist.set_clip_on(False)
    cax = fig.add_axes((477/600, 31/270, 38/600, 219/270))
    fig.colorbar(mappable, cax=cax)


def run(data_path, output_dir=None):
    """Execute the literal source computations and save figures in source order."""
    data_path = Path(data_path)
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != _DATA_SHA256:
        raise ValueError("AtmosphericData.mat must match Chebfun examples f4b9ea46")
    output_dir = Path(output_dir) if output_dir else _ROOT / "docs/images/sphere"
    output_dir.mkdir(parents=True, exist_ok=True)
    chebfun_style()
    figure_number = 0

    def save(fig):
        nonlocal figure_number
        figure_number += 1
        fig.set_facecolor("white")
        save_chebfun_figure(
            fig, output_dir / f"AtmosphericTemperature_{figure_number:02d}.png",
            size=(600, 270),
            layout="matlab" if figure_number in (2, 3, 4, 5, 6, 7, 8, 9, 10) else None,
        )
        plt.close(fig)

    def surface(fun, title="", *, colorbar=False, method="surf"):
        # Public mappable carries the exact surface data and color normalization.
        fig, ax, mappable = getattr(fun, method)(
            n_pts=200, cmap=_website_jet(), return_mappable=True,
        )
        ax.set_axis_off()
        matlab_view(ax, 50, 0)
        matlab_explicit_camera(ax, **_SOURCE_SURFACE_CAMERA)
        plot_earth(ax, "k-")
        if colorbar:
            _website_colorbar_layout(fig, ax, mappable)
        if title:
            ax.set_title(title)
        save(fig)

    # Native lines 28-44: Kelvin constructor, surface and object display.
    temperature = np.asarray(loadmat(data_path)["Temp"], dtype=float)
    f = Spherefun.from_values(temperature)
    surface(f, colorbar=True)
    _print_source_spherefun_display(f)

    # Native lines 59-70. Cartesian poles map exactly to these spherical angles.
    f = f - 273.15
    _print_source_ans(f.mean2())
    _print_source_ans(f(0.0, 0.0))
    _print_source_ans(f(0.0, np.pi))

    # Public MATLAB argument-stream adapter uses the native plotData grid.
    fig, ax = matlab_plot(f.slice_theta(np.pi / 2))
    ax.set_xlabel(r"Longitude, $\lambda$")
    ax.set_ylabel("Temperature (Celsius)")
    save(fig)

    fig, ax = f.contour(levels=np.arange(-40, 41, 5), n_pts=200, linewidth=2.0)
    ax.set_axis_off()
    matlab_view(ax, 50, 5)
    matlab_explicit_camera(ax, **_SOURCE_CONTOUR_CAMERA)
    plot_earth(ax, "k-")
    save(fig)

    zonal_mean = f.mean(dim=2)
    fig, ax = matlab_plot(zonal_mean)
    ax.set_xlim(0.0, np.pi)
    ax.set_xlabel(r"Co-latitude, $\theta$")
    ax.set_ylabel("Temperature (Celsius)")
    save(fig)

    # Requested sizes come from the Celsius function, not its resampled RHS.
    n, m = f.length()
    steady_heat = Spherefun.poisson(-(f - f.mean2()), 0, m, n)
    surface(f, "Original dataset", method="plot")
    surface(steady_heat, "Steady Heat", method="plot")

    sig = np.asarray([2, 10, 20]) * np.pi / 180
    surface(f, "Original Temp.")
    for sigma, degrees in zip(sig, (2, 10, 20), strict=True):
        fsmooth = f.gaussfilt(sigma)
        surface(fsmooth, rf"Smoothed Temp., $\sigma$={degrees} degrees")


def _resolve_data(data_path=None):
    """Retain the standalone cache contract with pinned, verified downloads."""
    if data_path is not None:
        return Path(data_path)
    cache = _ROOT / ".atmospheric_data.mat"
    if not cache.exists():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=_ROOT, prefix=".atmospheric_data.", suffix=".mat", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                with urllib.request.urlopen(_MAT_URL) as response:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != _DATA_SHA256:
                raise ValueError("AtmosphericData.mat must match Chebfun examples f4b9ea46")
            temporary.replace(cache)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if hashlib.sha256(cache.read_bytes()).hexdigest() != _DATA_SHA256:
        raise ValueError("AtmosphericData.mat must match Chebfun examples f4b9ea46")
    return cache


def main(argv=None):
    """Run the example with the original no-argument cache behavior."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.environ.get("ATMOSPHERIC_DATA"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    run(_resolve_data(args.data), args.output)


if __name__ == "__main__":
    main()
