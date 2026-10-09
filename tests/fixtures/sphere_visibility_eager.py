"""Frozen accepted visibility reference; base 4df8fffc."""

import jax.numpy as jnp


def _sphere_visible_vertices(points, projection, radius):
    """Test open sphere intersection along the point-to-camera ray segment.

    Matplotlib's get_proj matrices include world scaling and camera roll.
    Inverse column 2 is the perspective eye in homogeneous coordinates; for
    orthographic matrices its negative spatial part points toward the eye.
    Clamping the closest-point parameter includes interior starting points.
    Boundary comparisons use outward arithmetic bounds instead of a fitted
    epsilon. This remains analytic sphere geometry, not a mesh depth buffer.
    """
    points = jnp.asarray(points, dtype=jnp.float64)
    eye_h = jnp.linalg.inv(jnp.asarray(projection))[:, 2]
    if bool(eye_h[3] == 0):
        ray = jnp.broadcast_to(-eye_h[:3], points.shape)
        maximum = jnp.inf
    else:
        ray = eye_h[:3] / eye_h[3] - points
        maximum = 1.0
    norm2 = jnp.sum(ray * ray, axis=1)
    parameter = -jnp.sum(points * ray, axis=1) / jnp.where(norm2 > 0, norm2, 1.0)
    parameter = jnp.clip(parameter, 0.0, maximum)
    distance_upper = _sphere_ray_distance2_upper(points, ray, parameter)
    # Definite open-ball intersection only. An overlap between arithmetic
    # enclosures is a boundary ambiguity, so retain the source vertex.
    radius = jnp.asarray(radius, dtype=points.dtype)
    radius_lower = jnp.nextafter(radius * radius, -jnp.inf)
    occluded = distance_upper < radius_lower
    finite = jnp.all(jnp.isfinite(points), axis=1) & (norm2 > 0)
    return finite & ~occluded

def _sphere_ray_distance2_upper(points, ray, parameter):
    """Outward binary64 bound for ||points + parameter*ray|| squared.

    Each correctly rounded multiply/add lies between adjacent floats around
    its computed result. Propagate those intervals through the affine point,
    square their largest absolute endpoints, then bound the three products
    and two additions with gamma_5=5u/(1-5u). No camera uncertainty or
    polygonal-surface approximation is hidden in this arithmetic bound.
    """
    parameter = parameter[:, None]
    product = parameter * ray
    lower = jnp.nextafter(points + jnp.nextafter(product, -jnp.inf), -jnp.inf)
    upper = jnp.nextafter(points + jnp.nextafter(product, jnp.inf), jnp.inf)
    # Multiplication by zero and addition of signed zero are exact for our
    # finite real points/rays. Avoid charging an unnecessary affine error.
    lower = jnp.where(parameter == 0, points, lower)
    upper = jnp.where(parameter == 0, points, upper)
    extent = jnp.maximum(jnp.abs(lower), jnp.abs(upper))
    squared = extent * extent
    total = (squared[:, 0] + squared[:, 1]) + squared[:, 2]
    # Five operations: three products and two nonnegative additions. The
    # conservative gamma_5 relative forward bound remains valid if the
    # compiler fuses operations (which reduces their number). Round gamma
    # upward, its complementary denominator downward, and the result upward.
    unit_roundoff = jnp.finfo(points.dtype).eps / 2
    gamma = jnp.nextafter(5 * unit_roundoff / (1 - 5 * unit_roundoff), jnp.inf)
    denominator = jnp.nextafter(1 - gamma, -jnp.inf)
    return jnp.nextafter(total / denominator, jnp.inf)
