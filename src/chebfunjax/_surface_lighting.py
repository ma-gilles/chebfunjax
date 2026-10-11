"""Cartesian surface normals and local dull-material headlight.

Provenance
----------
MATLAB R2025b toolbox/matlab/graphics/graphics/graph3d/surfnorm.m,
material.m and camlight.m. Chebfun @ballfun/plot.m, commit 7574c77,
requests interpolated surface colors, headlight, phong and dull material.
This ports public SURFNORM arithmetic; equivalence to graphics automatic
normal generation and historical Phong rasterization is not established.
"""

import jax.numpy as jnp


def surface_vertex_normals(x, y, z):
    """Return unit Cartesian normals on a grid of at least 3 by 3 vertices.

    Provenance
    ----------
    Chebfun commit: 7574c77.
    MATLAB R2025b graph3d/surfnorm.m:61-109,132-137. Quadratic ghost
    extrapolation, centered stencils and cross-product orientation are
    preserved. Degenerate vertices produce zero normals, as the source does.
    """
    x, y, z = (jnp.asarray(a) for a in (x, y, z))
    if x.ndim != 2 or x.shape != y.shape or x.shape != z.shape:
        raise ValueError('surface coordinates require matching 2D grids')
    if min(x.shape) < 3:
        raise ValueError('surface normals require at least 3 by 3 vertices')

    def derivatives(v):
        v = jnp.concatenate(((3*v[0]-3*v[1]+v[2])[None], v,
                             (3*v[-1]-3*v[-2]+v[-3])[None]), axis=0)
        v = jnp.concatenate(((3*v[:, 0]-3*v[:, 1]+v[:, 2])[:, None], v,
                             (3*v[:, -1]-3*v[:, -2]+v[:, -3])[:, None]), axis=1)
        a = (v[1:-1, :-2]-v[1:-1, 2:])/2
        b = (v[2:, 1:-1]-v[:-2, 1:-1])/2
        return a, b

    ax, bx = derivatives(x)
    ay, by = derivatives(y)
    az, bz = derivatives(z)
    nx = -(ay*bz-az*by)
    ny = -(az*bx-ax*bz)
    nz = -(ax*by-ay*bx)
    magnitude = jnp.sqrt(nx*nx+ny*ny+nz*nz)
    magnitude = jnp.where(magnitude == 0, jnp.finfo(magnitude.dtype).eps, magnitude)
    return jnp.stack((nx/magnitude, ny/magnitude, nz/magnitude), axis=-1)


def dull_headlight_rgb(rgb, normals, x, y, z, camera_position, *, backface='lit'):
    """Apply a white local camera light and dull ambient/diffuse material.

    Provenance
    ----------
    Chebfun commit: 7574c77.
    MATLAB R2025b graph3d/material.m:120-123 sets dull ka=.3,kd=.8,ks=0.
    camlight.m:108-123 puts a local headlight at the camera position.
    MathWorks Surface Properties documents BackFaceLighting='reverselit':
    normals facing away from the camera are reversed. With a coincident
    local headlight this is abs(N dot L). The caller selects this policy
    explicitly; 'lit' retains oriented normals. White ambient/light colors,
    RGB clipping and vertex lighting are renderer assumptions, not a claim
    of exact MATLAB pixel or historical Phong rasterization parity.
    """
    if backface not in ('lit', 'reverselit'):
        raise ValueError("backface must be 'lit' or 'reverselit'")
    rgb, normals = jnp.asarray(rgb), jnp.asarray(normals)
    xyz = jnp.stack((jnp.asarray(x), jnp.asarray(y), jnp.asarray(z)), axis=-1)
    if rgb.shape != xyz.shape or normals.shape != xyz.shape:
        raise ValueError('RGB, normals and coordinates must have matching vertex shapes')
    camera_position = jnp.asarray(camera_position)
    if camera_position.shape != (3,):
        raise ValueError('camera_position requires three Cartesian coordinates')
    light = camera_position-xyz
    magnitude = jnp.sqrt(jnp.sum(light*light, axis=-1, keepdims=True))
    light = light/jnp.where(magnitude == 0, 1, magnitude)
    cosine = jnp.sum(normals*light, axis=-1)
    diffuse = jnp.abs(cosine) if backface == 'reverselit' else jnp.maximum(cosine, 0)
    return jnp.clip(rgb*(.3+.8*diffuse[..., None]), 0, 1)
