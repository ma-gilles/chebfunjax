"""plot_earth coastlines (MATLAB spherefun.plotEarth) (Fable 5).

Provenance
----------
MATLAB source : @spherefun/plotEarth.m
Chebfun commit: 7574c77
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from chebfunjax.plotting import plot_earth  # noqa: E402


def test_plot_earth_draws_unit_sphere_coastline():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    (line,) = plot_earth(ax, "k-")
    x, y, z = line.get_data_3d()
    r = np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2 + np.asarray(z) ** 2)
    assert len(x) == 8553
    assert np.nanmax(np.abs(r - 1)) < 1e-2
    plt.close(fig)
