"""Native public Helmholtz time stepping for sphere/SphereHeatConduction.m.

Original: https://www.chebfun.org/examples/sphere/SphereHeatConduction.html

Source: Chebfun examples f4b9ea46cfc2f52f20a844627f4a74d0bb10098c.
Copyright by The University of Oxford and The Chebfun Developers.
Plot colors/limits follow the cached chebfun.org HTML, which differs from
the example-repository .m file. Exact rendered visual parity remains open.
The rng(10) adapter below uses NumPy MT19937 uniform draws; same-version
MATLAB seed10 capture and complete page numerical/visual parity remain open.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import Formatter, FuncFormatter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style, matlab_explicit_camera, save_chebfun_figure
from chebfunjax.spherefun import Spherefun, spherefun

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'sphere')
FIG = [0]
ALPHA = 1 / 42
DT = .01
NSTEPS = 100


def _website_colormap():
    # R2017a graph3d/hot.m at the historical 64-color figure default.
    m = 64
    n = 3 * m // 8
    red = np.concatenate((np.arange(1, n+1)/n, np.ones(m-n)))
    green = np.concatenate((np.zeros(n), np.arange(1, n+1)/n, np.ones(m-2*n)))
    blue = np.concatenate((np.zeros(2*n), np.arange(1, m-2*n+1)/(m-2*n)))
    return ListedColormap(np.column_stack((red, green, blue))[::-1])


def _website_layout(fig, ax, mappable, title, has_contour):
    """Measured cached-PNG presentation, not MATLAB's opaque HG2 layout.

    All ten610x276 references locate the colorbar at x486..526, bottom31.
    The firstsix have top21; held-contour snapshots07..10 have top25.
    Their visible sphere diameters are145 and142pixels respectively. These
    presentation constants do not alter field values, color limits or mesh.
    Source surf view(3) fixes orientation(-37.5,30); projection isorthographic.
    """
    width, height = 610., 276.
    viewport_height = 220. if has_contour else 224.
    # Agg Gouraud edge rasterization extends the saturated footprint by
    # approximately one pixel per side; measured v2/reference bounds bind
    # this explicit presentation allowance, independently of field values.
    sphere_diameter = 140. if has_contour else 143.
    center_x = 244. if has_contour else 243.5
    fig.set_layout_engine(None)
    ax.set_position((0., 30.5/height, 2*center_x/width, viewport_height/height))
    # Explicit orthographic aperture follows unit-sphere pixel geometry.
    # It is an observable website presentation policy, not nativeauto-camera.
    distance = 10*np.sqrt(3.)
    azimuth, elevation = np.deg2rad(-127.5), np.deg2rad(30.)
    position = distance * np.array([np.cos(elevation)*np.cos(azimuth),
                                    np.cos(elevation)*np.sin(azimuth),
                                    np.sin(elevation)])
    fov = 2*viewport_height/sphere_diameter
    view_angle = np.rad2deg(2*np.arctan(fov/(2*distance)))
    matlab_explicit_camera(ax, position=position, view_angle=view_angle)
    cax = fig.add_axes((486./width, 31./height, 40./width, viewport_height/height))
    colorbar = fig.colorbar(mappable, cax=cax)
    colorbar.formatter = FuncFormatter(lambda value, _: Formatter.fix_minus(f"{value:g}"))
    colorbar.update_ticks()
    # Keep the genuine axes-title artist; place it in the measured figure band.
    if title:
        ax.title.set_transform(fig.transFigure)
        ax.title.set_position((center_x/width, 274./height))
        ax.title.set_verticalalignment("top")
        ax._autotitlepos = False


def _snap(field, clim, title="", mean_level=None):
    fig, ax, mappable = field.plot(title=title, cmap=_website_colormap(), clim=clim, return_mappable=True)
    if mean_level is not None:
        field.contour(ax=ax, hold=True, levels=[mean_level, mean_level], fmt="b-")
    _website_layout(fig, ax, mappable, title, mean_level is not None)
    ax.set_axis_off()
    FIG[0] += 1
    save_chebfun_figure(fig, os.path.join(_IMG, f"SphereHeatConduction_{FIG[0]:02d}.png"),
                       size=(610, 276))
    plt.close(fig)


def _trajectory(initial, m):
    """Yield the literal BDF1 bootstrap and all 99 BDF2 public solves."""
    yield 0, initial
    previous = initial
    k = np.sqrt(1 / (DT * ALPHA)) * 1j
    current = Spherefun.helmholtz(k**2 * previous, k, m, m)
    yield 1, current
    k = np.sqrt(3 / (2 * DT * ALPHA)) * 1j
    for n in range(2, NSTEPS + 1):
        rhs = k**2 / 3 * (4 * current - previous)
        previous = current
        current = Spherefun.helmholtz(rhs, k, m, m)
        yield n, current


def _run_initial(initial, m, clim, mean_level=None):
    for n, current in _trajectory(initial, m):
        if n == 0:
            _snap(current, clim)
        elif n % 25 == 0:
            _snap(current, clim, f"Time {n * DT:1.2f}", mean_level)
    return current


def _gaussian_initial():
    rng = np.random.RandomState(10)
    initial = Spherefun.empty()
    for _ in range(5):
        x0 = 2 * rng.random_sample() - 1
        y0 = np.sqrt(1 - x0**2) * (2 * rng.random_sample() - 1)
        z0 = np.sqrt(1 - x0**2 - y0**2)
        initial = initial + spherefun(
            lambda x, y, z: jnp.exp(-30 * ((x-x0)**2 + (y-y0)**2 + (z-z0)**2)))
    return initial


def run():
    os.makedirs(_IMG, exist_ok=True)
    initial = Spherefun.sphharm(6, 0) + np.sqrt(14 / 11) * Spherefun.sphharm(6, 5)
    final = _run_initial(initial, 20, (-1., 1.5))
    exact = np.exp(-42 * ALPHA * (NSTEPS * DT)) * initial
    print("ans =")
    print(f"     {float((final - exact).norm()):.15e}")

    initial = _gaussian_initial()
    mean_initial = float(initial.mean2())
    final = _run_initial(initial, 150, (-.05, 1.), mean_initial)
    print("ans =")
    print(f"     {abs(mean_initial - float(final.mean2())):.15e}")


if __name__ == "__main__":
    run()
