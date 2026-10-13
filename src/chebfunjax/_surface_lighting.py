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


def matlab_camlight_position(camera_position, camera_target, data_aspect_ratio,
                             camera_up_vector, position='left'):
    """Return MATLAB ``camlight`` position in data coordinates.

    Provenance
    ----------
    MATLAB source: R2024b ``camlight.m`` and ``camrotate.m`` camera-coordinate
    contract, checked against the retained R2024b ``camlight left`` and
    ``headlight`` property capture. Chebfun source: ``@separableApprox/plot.m``
    calls the surface plotting path; this helper only computes the light point.
    The implementation uses an orthonormal camera basis and the observed
    camera-relative angle, not copied MATLAB implementation code. Assumes
    right-handed axes and the supplied camera up vector is not parallel to
    the viewing direction.

    ``position`` is ``'left'``, ``'right'`` or ``'headlight'``. Local left and
    right lights lie on the sphere centered at the camera target with the same
    data-aspect-scaled radius as the camera. ``headlight`` returns the camera
    position itself.
    """
    if position not in ('left', 'right', 'headlight'):
        raise ValueError("position must be 'left', 'right', or 'headlight'")
    camera_position, camera_target, data_aspect_ratio, camera_up_vector = (
        jnp.asarray(value, dtype=jnp.float64)
        for value in (camera_position, camera_target, data_aspect_ratio,
                      camera_up_vector)
    )
    if any(value.shape != (3,) for value in
           (camera_position, camera_target, data_aspect_ratio, camera_up_vector)):
        raise ValueError('camera vectors and data aspect ratio must have shape (3,)')
    if not bool(jnp.all(data_aspect_ratio > 0)):
        raise ValueError('data aspect ratio must be positive')
    if position == 'headlight':
        return camera_position

    displacement = (camera_position-camera_target)/data_aspect_ratio
    distance = jnp.linalg.norm(displacement)
    if not bool(distance > 0):
        raise ValueError('camera position and target must differ')
    eye = displacement/distance
    up = camera_up_vector/data_aspect_ratio
    up = up-jnp.dot(up, eye)*eye
    up_norm = jnp.linalg.norm(up)
    if not bool(up_norm > 0):
        raise ValueError('camera up vector must not be parallel to viewing direction')
    up = up/up_norm
    right = jnp.cross(up, eye)
    right_norm = jnp.linalg.norm(right)
    if not bool(right_norm > 0):
        raise ValueError('camera basis is degenerate')
    right = right/right_norm

    # Native capture gives left-camera coordinates
    # (-sin(30)*cos(30), sin(30), cos(30)**2). Right mirrors the horizontal
    # coordinate. Scale back from data-aspect units to data coordinates.
    horizontal = jnp.sqrt(jnp.asarray(3., dtype=eye.dtype))/4
    direction = (eye*(3/4)+up/2
                 + right*((-1 if position == 'left' else 1)*horizontal))
    return camera_target+distance*direction*data_aspect_ratio


def matlab_camera_and_camlight_left(view_azimuth, view_elevation, target,
                                   data_aspect_ratio, radius):
    """Construct the camera and ``camlight left`` points in data coordinates.

    Provenance
    ----------
    MATLAB R2024b camlight/camera-coordinate semantics, verified against
    ``camlight left`` property captures at three views and nonuniform data
    aspect ratios. Chebfun source ``@separableApprox/surf.m`` then forwards
    the surface to MATLAB ``surf``. This is an independent basis derivation,
    not a port of MATLAB implementation code.

    Angles use MATLAB view convention: azimuth is measured from negative y.
    The radius is in the aspect-scaled plot box.
    """
    azimuth, elevation = jnp.deg2rad(jnp.asarray(
        [view_azimuth, view_elevation], dtype=jnp.float64))
    target = jnp.asarray(target, dtype=jnp.float64)
    dar = jnp.asarray(data_aspect_ratio, dtype=jnp.float64)
    eye = jnp.array([jnp.sin(azimuth)*jnp.cos(elevation),
                     -jnp.cos(azimuth)*jnp.cos(elevation),
                     jnp.sin(elevation)])
    right = jnp.array([jnp.cos(azimuth), jnp.sin(azimuth), 0.0])
    up = jnp.array([-jnp.sin(azimuth)*jnp.sin(elevation),
                    jnp.cos(azimuth)*jnp.sin(elevation),
                    jnp.cos(elevation)])
    # Captured local camlight-left direction in camera coordinates.
    light_direction = (-jnp.sqrt(3.0)/4.0*right + 0.5*up + 0.75*eye)
    camera = target + radius*dar*eye
    light = target + radius*dar*light_direction
    return camera, light


def matlab_flat_face_lighting(vertex_rgb, x, y, z, camera_position,
                              light_position, data_aspect_ratio, *,
                              ambient=0.3, diffuse=0.6, specular=0.9,
                              specular_exponent=10.0, backface='reverselit'):
    """Apply captured MATLAB flat-light material values to a sampled grid.

    Returns one RGBA value per grid quad. RGB inputs are the average of the
    four source-interpolated vertex colors, matching a flat face-color
    renderer adapter; this does not reproduce MATLAB's native rasterizer.

    Provenance
    ----------
    Chebfun commit 7574c77, ``@separableApprox/surf.m`` uses interpolated
    face colors with no edges. MATLAB R2024b Update 3 surface properties
    captured from that downstream call after ``camlight left``: flat lighting,
    ambient=.3, diffuse=.6, specular=.9, exponent=10, reversed back faces.
    The JAX computation is a local-light Blinn-Phong material evaluation;
    renderer interpolation, attenuation and pixel parity remain unqualified.
    """
    if backface not in ('lit', 'reverselit'):
        raise ValueError("backface must be 'lit' or 'reverselit'")
    x, y, z = (jnp.asarray(a, dtype=jnp.float64) for a in (x, y, z))
    rgb = jnp.asarray(vertex_rgb, dtype=jnp.float64)
    dar = jnp.asarray(data_aspect_ratio, dtype=jnp.float64)
    camera = jnp.asarray(camera_position, dtype=jnp.float64)
    light = jnp.asarray(light_position, dtype=jnp.float64)
    if x.ndim != 2 or x.shape != y.shape or x.shape != z.shape:
        raise ValueError('surface coordinates must have matching 2D shapes')
    if rgb.shape != x.shape + (4,):
        raise ValueError('vertex_rgb must have one RGBA value per grid vertex')
    if dar.shape != (3,) or camera.shape != (3,) or light.shape != (3,):
        raise ValueError('aspect ratio, camera and light must have shape (3,)')
    if min(x.shape) < 2:
        raise ValueError('surface grid must be at least 2 by 2')

    xyz = jnp.stack((x/dar[0], y/dar[1], z/dar[2]), axis=-1)
    p00, p01 = xyz[:-1, :-1], xyz[:-1, 1:]
    p10, p11 = xyz[1:, :-1], xyz[1:, 1:]
    normal = jnp.cross(p01-p00, p10-p00)
    normal_norm = jnp.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal/jnp.where(normal_norm == 0, 1.0, normal_norm)
    center = (p00+p01+p10+p11)*0.25
    camera_scaled, light_scaled = camera/dar, light/dar
    eye = camera_scaled-center
    to_light = light_scaled-center
    eye = eye/jnp.maximum(jnp.linalg.norm(eye, axis=-1, keepdims=True),
                           jnp.finfo(eye.dtype).tiny)
    to_light = to_light/jnp.maximum(
        jnp.linalg.norm(to_light, axis=-1, keepdims=True),
        jnp.finfo(to_light.dtype).tiny)
    if backface == 'reverselit':
        normal = jnp.where(jnp.sum(normal*eye, axis=-1, keepdims=True) < 0,
                           -normal, normal)
    ndotl = jnp.maximum(jnp.sum(normal*to_light, axis=-1), 0.0)
    half_vector = eye+to_light
    half_norm = jnp.linalg.norm(half_vector, axis=-1, keepdims=True)
    half_vector = half_vector/jnp.maximum(half_norm,
                                          jnp.finfo(half_vector.dtype).tiny)
    ndoth = jnp.maximum(jnp.sum(normal*half_vector, axis=-1), 0.0)
    highlight = ndoth**specular_exponent
    base = (rgb[:-1, :-1]+rgb[:-1, 1:]+rgb[1:, :-1]+rgb[1:, 1:])*0.25
    shaded = base[..., :3]*(ambient+diffuse*ndotl[..., None])
    shaded = jnp.clip(shaded+specular*highlight[..., None], 0.0, 1.0)
    alpha = base[..., 3:4]
    return jnp.concatenate((shaded, alpha), axis=-1)
