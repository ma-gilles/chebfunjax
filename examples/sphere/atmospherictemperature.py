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
from scipy.io import loadmat

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from chebfunjax.plotting import (  # noqa: E402
    chebfun_style,
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


def _print_source_ans(value):
    """Computed scalar output; MATLAB session formatting is not inferred."""
    print("\nans =\n")
    print(f"{float(value):20.15f}\n")


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
        )
        plt.close(fig)

    def surface(fun, title="", *, colorbar=False, method="surf"):
        # Public mappable carries the exact surface data and color normalization.
        fig, ax, mappable = getattr(fun, method)(
            n_pts=200, cmap="jet", return_mappable=True,
        )
        if colorbar:
            fig.colorbar(mappable, ax=ax)
        ax.set_axis_off()
        matlab_view(ax, 50, 0)
        plot_earth(ax, "k-")
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
