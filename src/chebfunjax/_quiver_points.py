"""Numeric point quiver using readable MATLAB scaling and compatibility lines."""

import jax
import jax.numpy as jnp


def quiver_points(x, y, u, v, ax=None, autoscale_factor=0.9, **kw):
    """Plot finite, equal-shaped real vectors or matrices; return ``(fig, ax)``.

    A one-dimensional Python array represents a MATLAB row vector. The fitting
    factor defaults to .9; zero disables fitting. LineCollection properties and
    ``max_head_size``/``show_arrow_head`` are supported. Heads are open lines.

    Scaling and geometry follow R2025b createLinesForQuiverStruct.m, also used
    by the Chebfun 7574c77 quiver wrappers. This is readable compatibility
    geometry, not opaque R2025b renderer or pixel equivalence. Existing manual
    axis limits and aspect are preserved. Automatic limits use Matplotlib's
    policy; MATLAB automatic limit rounding is not implemented here.

    Provenance
    ----------
    MATLAB wrapper source: @chebfun2v/quiver.m.
    Native numeric scaling: MATLAB R2025b createLinesForQuiverStruct.m.
    Readable compatibility heads reuse chebfunjax._quiver_geometry.
    Chebfun commit: 7574c77.
    """
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection

    from chebfunjax._quiver_geometry import quiver_line_geometry
    from chebfunjax.plotting import (
        _quiver_finite_data,
        _quiver_head_options,
        _quiver_line_options,
        _quiver_options,
        _quiver_source_scale,
    )

    x, y, u, v = _quiver_finite_data(x, y, u, v)
    if x.ndim not in (1, 2) or x.size == 0:
        raise ValueError("quiver_points requires nonempty vectors or matrices")
    source, factor = _quiver_options(autoscale_factor, kw)
    if not source:
        raise ValueError("quiver_points requires source scaling and line properties")
    if x.ndim == 1:
        x, y, u, v = (a.reshape(1, -1) for a in (x, y, u, v))
    # Float64 avoids integer overflow in squared lengths and matches double
    # point inputs of the native examples.
    x, y, u, v = (jnp.asarray(a, dtype=jnp.float64) for a in (x, y, u, v))
    scale = _quiver_source_scale(x, y, u, v, factor)
    size, show_head = _quiver_head_options(kw)
    shafts, heads, _ = quiver_line_geometry(x, y, u * scale, v * scale, size)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    opts = _quiver_line_options(ax, kw)
    # Host transfers only hand completed JAX geometry to the renderer.
    ax.add_collection(LineCollection(jax.device_get(shafts), **opts))
    opts = dict(opts, label="_nolegend_")
    opts["visible"] = opts.get("visible", True) and show_head
    ax.add_collection(LineCollection(jax.device_get(heads), **opts))
    ax.autoscale_view()
    return fig, ax
