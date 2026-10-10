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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style, save_chebfun_figure
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


def _snap(field, clim, title="", mean_level=None):
    fig, ax, mappable = field.plot(title=title, cmap=_website_colormap(), clim=clim, return_mappable=True)
    if mean_level is not None:
        field.contour(ax=ax, hold=True, levels=[mean_level, mean_level], fmt="k-")
    fig.colorbar(mappable, ax=ax)
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
