# uses-numpy: Independent geometric coordinates and artist property assertions.
"""Source @spherefun/contour.m carries LineWidth/LineStyle/Color to plot3.

The equator of f=z is an independent geometric control; these checks do
not qualify MATLAB pixel rendering or all contour options.
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from chebfunjax.spherefun.spherefun import Spherefun


@pytest.mark.parametrize("fmt", [None, "r--"])
def test_explicit_line_properties_on_equatorial_contour(fmt):
    f = Spherefun.from_function(lambda lam, theta: jnp.cos(theta))
    fig, ax = f.contour(levels=[0.0, 0.0], n_pts=24, fmt=fmt,
                        linewidth=2.0, linestyle=":", color="purple", alpha=0.4)
    try:
        assert len(ax.lines) > 0
        for line in ax.lines:
            assert line.get_linewidth() == 2.0
            assert line.get_linestyle() == ":"
            assert line.get_color() == "purple"
            assert line.get_alpha() == 0.4
            x, y, z = map(np.asarray, line.get_data_3d())
            np.testing.assert_allclose(z, 0.0, rtol=0, atol=2e-15)
            np.testing.assert_allclose(x*x+y*y+z*z, 1.0, rtol=0,
                                       atol=8*np.finfo(float).eps)
    finally:
        plt.close(fig)
