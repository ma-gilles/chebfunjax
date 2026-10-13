"""Source-shaped graphics methods for dense Chebfun3T tensors.

Provenance
----------
MATLAB source: ``@chebfun3t/plotcoeffs.m``
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Original authors: Copyright 2017 by The University of Oxford and the
Chebfun Developers.

The coefficient and color-data calculations use JAX. Conversion to NumPy
occurs only when handing concrete arrays to Matplotlib's renderer.
"""

from __future__ import annotations


class Chebfun3TPlotMixin:
    """MATLAB-shaped 3-D coefficient plot for a dense tensor."""

    coeffs: object
    vscale: object

    def plotcoeffs(self, ax=None):
        """Scatter the logarithmic coefficient magnitudes on a 3-D index grid.

        The source makes ``scatter3`` and colorbar output available, but its
        requested-output branch returns an undefined variable ``h``. This
        Python method returns ``(fig, ax)`` as an explicit renderer adapter;
        it does not claim that the source output branch succeeds.

        Provenance
        ----------
        MATLAB source: ``@chebfun3t/plotcoeffs.m``
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        import jax.numpy as jnp
        import matplotlib.pyplot as plt
        import numpy as np  # uses-numpy: concrete Matplotlib renderer boundary

        if ax is None:
            if self.isempty():
                fig, ax = plt.subplots()
                ax.plot([])
                return fig, ax
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
        else:
            fig = ax.get_figure()
            if self.isempty():
                ax.plot([])
                return fig, ax

        if not hasattr(ax, "view_init"):
            raise TypeError("Chebfun3T.plotcoeffs requires a 3-D Matplotlib axes")

        magnitudes = jnp.abs(jnp.asarray(self.coeffs))
        if not jnp.issubdtype(magnitudes.dtype, jnp.floating):
            magnitudes = magnitudes.astype(jnp.float64)
        machine_epsilon = jnp.asarray(jnp.finfo(jnp.float64).eps, dtype=magnitudes.dtype)
        vscale = jnp.asarray(self.vscale)
        if bool(vscale > 0):
            minimum_nonzero = jnp.min(jnp.where(magnitudes > 0, magnitudes, jnp.inf))
            floor = jnp.minimum(machine_epsilon * vscale, minimum_nonzero)
            magnitudes = jnp.where(magnitudes == 0, floor, magnitudes)
        else:
            magnitudes = magnitudes + machine_epsilon

        m, n, p = (int(size) for size in magnitudes.shape)
        i, j, k = (jnp.arange(1, size + 1) for size in (m, n, p))
        x, y, z = jnp.meshgrid(i, j, k, indexing="ij")

        # MATLAB ndgrid followed by (:), which is column-major/F order.
        x = np.asarray(jnp.ravel(x, order="F"))
        y = np.asarray(jnp.ravel(y, order="F"))
        z = np.asarray(jnp.ravel(z, order="F"))
        log_magnitudes = np.asarray(jnp.ravel(jnp.log10(magnitudes), order="F"))

        from chebfunjax.plotting import PARULA, matlab_view

        artist = ax.scatter(
            x,
            y,
            z,
            s=36,
            c=log_magnitudes,
            cmap=PARULA,
            marker="o",
            edgecolors="none",
            linewidths=0.5,
            depthshade=False,
        )
        ax.grid(True)
        fig.colorbar(artist, ax=ax)
        matlab_view(ax, -30, 35)
        return fig, ax
