# uses-numpy: matplotlib rendering is not JIT-safe
"""Phase portrait visualisation for complex-valued functions.

Translated from MATLAB Chebfun (commit 7574c77): phaseplot.m.
Original: Copyright 2020 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

References
----------
[1] E. Wegert, *Visual Complex Functions: An Introduction with Phase Portraits*,
    Springer Basel, 2012.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

# ===========================================================================
# Public API
# ===========================================================================


def phaseplot(
    f: Callable,
    ax: Optional[Sequence[float]] = None,
    *,
    classic: bool = False,
    caxis_start: float = -np.pi,
    n_pts: int = 500,
    colormap: str = "hsv",
) -> "np.ndarray":
    """Phase (argument) plot of a complex-valued function.

    Draws a phase plot of f(z) in the complex plane using an HSV-based
    colour scheme.  As arg(f(z)) ranges over [-pi, pi] the colours cycle
    cyan -> blue -> magenta -> red -> yellow -> green.

    Parameters
    ----------
    f : callable
        Complex-valued function of a complex variable.  Must accept a 2-D
        complex numpy array and return an array of the same shape.
    ax : sequence of 4 floats, optional
        Axes limits ``[x_min, x_max, y_min, y_max]``.  Defaults to
        ``[-1, 1, -1, 1]``.
    classic : bool, optional
        If True, use the classic phase colouring (arg directly -> hue) as
        in Wegert [1].  If False (default) use a slightly smoothed variant
        that is less harsh:
            phi(t) = t - 0.5 * cos(1.5*t)^3 * sin(1.5*t)
    caxis_start : float, optional
        Start of the colour axis (default -pi).  The colour axis always
        spans exactly 2*pi.
    n_pts : int, optional
        Number of grid points in each direction (default 500).
    colormap : str, optional
        Matplotlib colormap name for the HSV palette (default ``'hsv'``).

    Returns
    -------
    img : np.ndarray, shape (n_pts, n_pts, 3), dtype float
        RGB image array suitable for ``matplotlib.pyplot.imshow``.  The
        (0, 0) corner corresponds to (ax[0], ax[3]) — upper left in the
        standard image orientation.

    Notes
    -----
    This function returns the raw RGB array.  To display it::

        import matplotlib.pyplot as plt
        img = phaseplot(f, [-2, 2, -2, 2])
        plt.imshow(img, extent=[-2, 2, -2, 2], origin='lower')
        plt.show()

    Provenance
    ----------
    MATLAB source : phaseplot.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2020 by The University of Oxford
        and The Chebfun Developers.

    Examples
    --------
    Phase plot of z^2:

    >>> import numpy as np
    >>> img = phaseplot(lambda z: z**2, [-2, 2, -2, 2])
    >>> img.shape
    (500, 500, 3)
    >>> img.dtype
    dtype('float64')

    See Also
    --------
    conformal, conformal2
    """
    if ax is None:
        ax = [-1.0, 1.0, -1.0, 1.0]
    ax = list(ax)
    x_min, x_max, y_min, y_max = ax

    x = np.linspace(x_min, x_max, n_pts)
    y = np.linspace(y_min, y_max, n_pts)
    xx, yy = np.meshgrid(x, y)
    zz = xx + 1j * yy

    # Evaluate f on the grid
    fz = np.asarray(f(zz), dtype=complex)

    # Compute phase angle
    phase = np.angle(fz)  # in [-pi, pi]

    # Apply smoothing transformation if not classic
    theta_start = float(caxis_start)
    if classic:
        phi = phase
    else:
        phi = phase - 0.5 * np.cos(1.5 * phase) ** 3 * np.sin(1.5 * phase)

    # Map to [0, 1] for colourmap lookup
    # MATLAB: mod(phi(angle(f(zz)))-pi+pi-theta, 2*pi)+theta, normalised to [0,1]
    hue = np.mod(phi - theta_start, 2.0 * np.pi) / (2.0 * np.pi)  # in [0, 1)

    # Build RGB image using matplotlib colormap
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap, 600)
        # Shift the colourmap if theta_start != 0
        shift = int(round(600 * np.mod(theta_start, 2.0 * np.pi) / (2.0 * np.pi)))
        rgba = cmap(hue)
        img = rgba[:, :, :3]  # drop alpha

        if shift != 0:
            # Recreate cyclic colormap with shift baked in
            colors = cmap(np.linspace(0, 1, 600, endpoint=False))
            colors = np.roll(colors, -shift, axis=0)
            from matplotlib.colors import ListedColormap
            cmap_shifted = ListedColormap(colors)
            img = cmap_shifted(hue)[:, :, :3]
    except ImportError:
        # Fallback: HSV to RGB manually
        img = _hsv_to_rgb(hue)

    # Flip vertically so y_min is at bottom (origin='lower' convention)
    img = img[::-1, :, :]
    return img


# ===========================================================================
# Private helpers
# ===========================================================================


def _hsv_to_rgb(h: np.ndarray) -> np.ndarray:
    """Convert hue (S=V=1) to RGB.  h in [0, 1)."""
    h = h * 6.0
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)

    r = np.ones_like(h)
    g = np.ones_like(h)
    b = np.ones_like(h)

    for idx, (r_val, g_val, b_val) in enumerate([
        (1, f, 0),          # 0 <= h < 1: red -> yellow
        (1 - f, 1, 0),      # 1 <= h < 2: yellow -> green
        (0, 1, f),          # 2 <= h < 3: green -> cyan
        (0, 1 - f, 1),      # 3 <= h < 4: cyan -> blue
        (f, 0, 1),          # 4 <= h < 5: blue -> magenta
        (1, 0, 1 - f),      # 5 <= h < 6: magenta -> red
    ]):
        mask = i == idx
        r[mask] = r_val if np.isscalar(r_val) else r_val[mask]
        g[mask] = g_val if np.isscalar(g_val) else g_val[mask]
        b[mask] = b_val if np.isscalar(b_val) else b_val[mask]

    return np.stack([r, g, b], axis=-1)
