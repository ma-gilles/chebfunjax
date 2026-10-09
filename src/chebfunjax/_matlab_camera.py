"""Explicit orthographic camera mapping; no inferred native auto aperture.

Licensed MATLAB R2017a camzoom.m supplies FOV = 2*distance*tan(CVA/2)
in data-aspect units; viewmtx.m supplies the orthographic camera convention.
The opt-in adapter targets sphere axes with an explicit captured camera.
It does not reconstruct MATLAB's internal automatic camera/layout engine.
"""
from __future__ import annotations

import math
from types import MethodType

import jax.numpy as jnp
import numpy as np  # uses-numpy: final Matplotlib projection-matrix interop only.
from mpl_toolkits.mplot3d import Axes3D


def _explicit_camera_projection(ax):
    state = ax._matlab_explicit_camera
    # Preserve Matplotlib's view-axis bookkeeping and user opt-out behavior.
    original = state["original_get_proj"]()
    current_view = (ax.elev, ax.azim, ax.roll)
    if (current_view != state["view"] or ax._focal_length != math.inf
            or tuple(ax.get_box_aspect()) != state["box_aspect"]):
        return original
    width, height = ax.bbox.width, ax.bbox.height
    if width <= 0 or height <= 0:
        raise ValueError("Explicit camera requires a positive viewport")
    # Matplotlib maps its pseudo-coordinates through viewLim to bbox pixels.
    # Map physical camera-plane coordinates to those coordinates, correcting
    # both its appearance-tuned aperture and asymmetric pseudo-view center.
    x0, y0, xspan, yspan = ax.viewLim.bounds
    xscale = xspan * height / (width * state["fov"])
    yscale = yspan / state["fov"]
    rows = state["camera_rows"]
    target = state["target"]
    xrow = xscale * rows[0]
    yrow = yscale * rows[1]
    zrow = -rows[2] / state["distance"]
    matrix = jnp.eye(4, dtype=jnp.float64)
    matrix = matrix.at[0, :3].set(xrow)
    matrix = matrix.at[1, :3].set(yrow)
    matrix = matrix.at[2, :3].set(zrow)
    matrix = matrix.at[0, 3].set(x0 + xspan/2 - jnp.dot(xrow, target))
    matrix = matrix.at[1, 3].set(y0 + yspan/2 - jnp.dot(yrow, target))
    matrix = matrix.at[2, 3].set(1 - jnp.dot(zrow, target))
    # Depth decreases toward the eye, matching Axes3D orthographic ordering.
    # Inverse column2 therefore retains the coast adapter's ray convention.
    return np.asarray(matrix)


def matlab_explicit_camera(ax, *, position, target=(0., 0., 0.),
                  up=(0., 0., 1.), view_angle, data_aspect=(1., 1., 1.)):
    """Apply an explicit orthographic source camera to this axes only.

    Camera values are in data coordinates; ``view_angle`` is degrees.
    Captured unit-sphere views50/0 and50/5 are the initial qualified target.
    This is a manual camera adapter, not a native automatic-camera policy.
    Layout/font/DPI/data/limits are untouched. Viewport scale is recalculated
    after draws/resizes. A subsequent explicit view/projection/box zoom change
    opts out to the original Matplotlib projection; reapply to set a camera.
    Existing plot_sphere supplied axes and matlab_view remain unchanged.
    """
    if not isinstance(ax, Axes3D):
        raise TypeError("Explicit MATLAB camera requires a 3D axes")
    fields = [tuple(float(x) for x in item) for item in (position, target, up, data_aspect)]
    if (any(len(item) != 3 for item in fields)
            or not all(math.isfinite(x) for item in fields for x in item)
            or any(x <= 0 for x in fields[3])):
        raise ValueError("Camera vectors require three finite values and positive data aspect")
    angle = float(view_angle)
    if not math.isfinite(angle) or not 0 < angle < 180:
        raise ValueError("Camera view angle must be finite and between 0 and 180 degrees")
    position, target, up, aspect = (jnp.asarray(item, dtype=jnp.float64) for item in fields)
    displacement = (position-target)/aspect
    distance = jnp.linalg.norm(displacement)
    if not bool(distance > 0):
        raise ValueError("Camera position and target must differ")
    backward = displacement/distance
    right = jnp.cross(up/aspect, backward)
    magnitude = jnp.linalg.norm(right)
    if not bool(magnitude > 0):
        raise ValueError("Camera up vector must not be parallel to the viewing direction")
    right = right/magnitude
    vertical = jnp.cross(backward, right)
    rows = jnp.stack((right, vertical, backward))/aspect[None, :]
    fov = 2*distance*jnp.tan(jnp.deg2rad(jnp.asarray(angle))/2)
    old = getattr(ax, "_matlab_explicit_camera", None)
    original = ax.get_proj if old is None else old["original_get_proj"]
    # Explicit camera position determines the view. This is opt-in and does
    # not alter the separate angle-only matlab_view contract.
    elev = float(jnp.rad2deg(jnp.arcsin(backward[2])))
    azim = float(jnp.rad2deg(jnp.arctan2(backward[1], backward[0])))
    ax.view_init(elev=elev, azim=azim, roll=0.)
    ax.set_proj_type("ortho")
    ax._matlab_explicit_camera = {
        "original_get_proj": original, "camera_rows": rows, "target": target,
        "distance": float(distance), "fov": float(fov),
        "view": (ax.elev, ax.azim, ax.roll),
        "box_aspect": tuple(ax.get_box_aspect()),
    }
    ax.get_proj = MethodType(_explicit_camera_projection, ax)
    ax.stale = True
    return {"projection": "orthographic", "fov": float(fov),
            "distance": float(distance), "policy": "explicit camera, no auto aperture"}
