"""Opt-in publication layout from MATLAB outer-position/inset properties.

This is a renderer policy adapter, not the proprietary MATLAB layout engine.
Native evidence: sphere_graphics_metadata_capture_v2_20261006 metadata.json,
SHA2563754684ed299c6bfb597114ef5e22a3739ebcf00da35a31dbcf35866ac96c8bc.
R2025b/7574c77 recorded normalized axes units, PositionConstraint outerposition
and LooseInset[.13,.11,.095,.075]. Insets are figure coordinates, not fractions
of the allocated outer rectangle. Historical typography is not established.
"""
from __future__ import annotations

import math

from matplotlib.layout_engine import PlaceHolderLayoutEngine


def matlab_axes_layout(ax, *, outer_position=(0., 0., 1., 1.),
                       loose_inset=(.13, .11, .095, .075)):
    """Apply source-derived outer-position/inset policy at current figure DPI.

    Only a single axes without colorbars, inset/twin axes, legends or an active
    layout engine is supported. Geometry, cameras, limits, fonts and locators
    remain untouched. Four geometry passes bound nonconvergent/infeasible cases;
    that guard is not a claim about MATLAB's internal iteration count.

    Both rectangles and insets use normalized parent-figure coordinates.
    Returns a diagnostic mapping of the measured/effective insets and position.
    """
    fig = ax.get_figure()
    engine = fig.get_layout_engine()
    if (len(fig.axes) != 1 or fig.axes[0] is not ax or ax.child_axes
            or getattr(ax, "_colorbar", None) is not None
            or ax.get_legend() is not None
            or (engine is not None and not isinstance(engine, PlaceHolderLayoutEngine))):
        raise ValueError("matlab layout supports one axes without colorbar, legend, inset/twin axes or active layout engine")
    outer = tuple(float(x) for x in outer_position)
    loose = tuple(float(x) for x in loose_inset)
    if (len(outer) != 4 or len(loose) != 4
            or not all(math.isfinite(x) for x in outer+loose)
            or outer[2] <= 0 or outer[3] <= 0
            or min(loose) < 0):
        raise ValueError("matlab layout requires finite outer rectangle and nonnegative figure-normalized insets")
    previous = ax.get_position(original=True).frozen()
    in_layout = ax.get_in_layout()
    width, height = float(fig.bbox.width), float(fig.bbox.height)
    renderer = fig._get_renderer()
    margins = loose
    history = []
    # Roundoff only, scaled to normalized figure coordinates. No pixel-fit slack.
    tolerance = 64*math.ulp(max(1., *map(abs, outer), *map(abs, loose)))
    try:
        for iteration in range(4):
            left, bottom, right, top = margins
            position = (outer[0]+left, outer[1]+bottom,
                        outer[2]-left-right, outer[3]-bottom-top)
            if position[2] <= 0 or position[3] <= 0:
                raise ValueError("matlab layout is infeasible at the requested canvas/font sizes")
            ax.set_position(position, which="both")
            ax.set_in_layout(in_layout)
            box = ax.get_tightbbox(renderer, bbox_extra_artists=[], for_layout_only=False)
            if box is None:
                raise ValueError("matlab layout requires a visible axes")
            extents = (float(box.x0)/width, float(box.y0)/height,
                       float(box.x1)/width, float(box.y1)/height)
            tight = (max(0., position[0]-extents[0]),
                     max(0., position[1]-extents[1]),
                     max(0., extents[2]-position[0]-position[2]),
                     max(0., extents[3]-position[1]-position[3]))
            updated = tuple(max(a, b) for a, b in zip(loose, tight))
            history.append({"position": position, "tight_inset": tight,
                            "effective_inset": margins, "decoration_bounds": extents})
            fits = (extents[0] >= outer[0]-tolerance
                    and extents[1] >= outer[1]-tolerance
                    and extents[2] <= outer[0]+outer[2]+tolerance
                    and extents[3] <= outer[1]+outer[3]+tolerance)
            if fits and max(abs(a-b) for a, b in zip(updated, margins)) <= tolerance:
                return {"outer_position": outer, "loose_inset": loose,
                        "position": position, "tight_inset": tight,
                        "effective_inset": margins, "passes": iteration+1,
                        "history": history, "units": "normalized figure"}
            margins = updated
        raise ValueError("matlab layout did not converge within four geometry passes")
    except Exception:
        ax.set_position(previous, which="both")
        ax.set_in_layout(in_layout)
        raise
