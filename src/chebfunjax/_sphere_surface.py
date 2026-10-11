"""Vertex-color interpolation for source sphere surfaces.

Source @spherefun/surf.m7574c77 requests FaceColor interp, EdgeColor none.
Matplotlib plot_surface paints a constant color per polygon. This adapter
retains the input vertices and RGBA, splits each quad along a fixed diagonal,
and submits depth-ordered Gouraud triangles to the output renderer. Native
OpenGL quad tessellation and pixel rasterization are not claimed identical.
"""
import numpy as np  # uses-numpy: final renderer projection/artist interop only.
from matplotlib.artist import allow_rasterization
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class InterpolatedSphereSurface(Poly3DCollection):
    """Opaque or uniformly translucent vertex-colored gridded surface."""

    def __init__(self, x, y, z, rgba, **kwargs):
        x, y, z, rgba = (np.asarray(a) for a in (x, y, z, rgba))
        if x.ndim != 2 or x.shape != y.shape or x.shape != z.shape:
            raise ValueError("surface coordinates require matching two-dimensional grids")
        if min(x.shape) < 2 or rgba.shape != x.shape + (4,):
            raise ValueError("surface requires at least2x2 vertices and oneRGBA pervertex")
        vertices = np.stack((x, y, z), axis=-1)
        ids = np.arange(x.size).reshape(x.shape)
        a, b, c, d = (v.ravel() for v in
                      (ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]))
        indices = np.concatenate((np.stack((a, b, c), axis=-1),
                                  np.stack((a, c, d), axis=-1)))
        self._source_vertices = vertices.copy()
        self._source_rgba = rgba.copy()
        self._triangle_indices = indices
        self._triangle_xyz = vertices.reshape(-1, 3)[indices]
        self._triangle_rgba = rgba.reshape(-1, 4)[indices]
        super().__init__([], edgecolors="none", **kwargs)

    def do_3d_projection(self):
        xyz = self._triangle_xyz.reshape(-1, 3)
        x, y, z = proj3d.proj_transform(xyz[:, 0], xyz[:, 1], xyz[:, 2], self.axes.M)
        xy = np.stack((x, y), axis=-1).reshape(-1, 3, 2)
        depth = z.reshape(-1, 3).mean(axis=1)
        order = np.argsort(-depth, kind="stable")
        self._projected_triangles = xy[order]
        self._projected_colors = self._triangle_rgba[order]
        # Retain collection color introspection used by ordinary Matplotlib.
        self._facecolors2d = self._projected_colors.mean(axis=1)
        self._edgecolors2d = np.empty((0, 4))
        return float(np.min(z))

    @allow_rasterization
    def draw(self, renderer):
        if not self.get_visible():
            return
        self.do_3d_projection()
        colors = self._projected_colors
        if self.get_alpha() is not None:
            colors = colors.copy()
            colors[..., 3] *= self.get_alpha()
        gc = renderer.new_gc()
        try:
            self._set_gc_clip(gc)
            renderer.draw_gouraud_triangles(gc, self._projected_triangles,
                                            colors, self.axes.transData)
        finally:
            gc.restore()
        self.stale = False


class InterpolatedSurfaceGroup(InterpolatedSphereSurface):
    """Order triangles from intersecting surface grids in one collection.

    Each input is an ``(x, y, z, rgba)`` tuple. Vertex colors and triangulation
    are preserved independently for each grid; normals are never averaged
    across intersections. Mean-depth ordering remains a painter algorithm,
    so intersecting triangles do not have native OpenGL pixel visibility.

    Provenance
    ----------
    MATLAB source: @ballfun/plot.m, plotBall surface loop.
    Chebfun commit: 7574c77.
    """

    def __init__(self, surfaces, **kwargs):
        surfaces = tuple(surfaces)
        if not surfaces:
            raise ValueError("at least one surface grid is required")
        super().__init__(*surfaces[0], **kwargs)
        parts = [self, *(InterpolatedSphereSurface(*item)
                         for item in surfaces[1:])]
        self._source_surfaces = tuple((part._source_vertices,
                                       part._source_rgba) for part in parts)
        self._triangle_xyz = np.concatenate([part._triangle_xyz for part in parts])
        self._triangle_rgba = np.concatenate([part._triangle_rgba for part in parts])
