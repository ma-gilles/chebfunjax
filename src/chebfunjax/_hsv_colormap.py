"""Explicit-length MATLAB HSV palette for renderer use.

Provenance
----------
MATLAB R2024b toolbox/matlab/graphics/graphics/color/hsv.m and
 toolbox/matlab/images/hsv2rgb.m.  The explicit-length branch computes
h=(0:m-1)'/max(m,1), then hsv2rgb([h ones(m,2)]).  This private candidate
keeps the palette independent of global renderer defaults.
"""

import operator

import jax.numpy as jnp


def hsv_colors(m):
    """Return the source HSV palette as an ``(m, 3)`` JAX array.

    Integer-like lengths follow ``operator.index`` like the existing explicit
    palette helpers.  The empty source map is ``(0, 3)``; for ``m == 1`` the
    source hue is zero and the sole color is red.

    Provenance
    ----------
    MATLAB R2024b graphics/graphics/color/hsv.m and images/hsv2rgb.m.
    Called with length 600 by @separableApprox/plot.m, Chebfun commit
    7574c77680d7e82b79626300bf255498271a72df.
    """
    m = operator.index(m)
    if m < 0:
        raise ValueError('colormap length must be nonnegative')
    if m == 0:
        return jnp.zeros((0, 3), dtype=jnp.float64)

    hue = jnp.arange(m, dtype=jnp.float64) / max(m, 1)
    h = 6.0 * hue
    k = jnp.floor(h).astype(jnp.int32)
    fraction = h - k
    n = 1.0 - fraction

    # MATLAB hsv2rgb's six sector assignments for S=V=1.  h never reaches 6
    # for this palette, so its k==6 wrap branch is not needed here.
    red = (k == 0).astype(jnp.float64)
    red += (k == 1).astype(jnp.float64) * n
    red += (k == 4).astype(jnp.float64) * fraction
    red += (k == 5).astype(jnp.float64)
    green = (k == 0).astype(jnp.float64) * fraction
    green += (k == 1).astype(jnp.float64)
    green += (k == 2).astype(jnp.float64)
    green += (k == 3).astype(jnp.float64) * n
    blue = (k == 2).astype(jnp.float64) * fraction
    blue += (k == 3).astype(jnp.float64)
    blue += (k == 4).astype(jnp.float64)
    blue += (k == 5).astype(jnp.float64) * n
    # Sector formulas above specialize the R2024b HSV-to-RGB equations to
    # saturation/value one while retaining the source's fractional hue ramp.
    return jnp.stack((red, green, blue), axis=1)


def hsv_colormap(m):
    """Build a ListedColormap with exactly ``m`` source HSV colors.

    Provenance
    ----------
    Matplotlib adapter for :func:`hsv_colors`, derived from MATLAB R2024b
    hsv.m/hsv2rgb.m; @separableApprox/plot.m, Chebfun commit
    7574c77680d7e82b79626300bf255498271a72df, requests hsv(600).
    """
    from matplotlib.colors import ListedColormap

    m = operator.index(m)
    if m < 1:
        raise ValueError('renderer colormap length must be positive')
    return ListedColormap(hsv_colors(m).tolist(), name=f'hsv_r2024b_{m}', N=m)
