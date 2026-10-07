"""Readable MATLAB compatibility geometry; opaque R2025b rendering unqualified."""
import jax.numpy as jnp


def quiver_line_geometry(x, y, u, v, max_head_size=0.2, z=None, w=None):
    """Return shaft/head/base arrays from expanded, already-scaled 2D or 3D data.

    Provenance
    ----------
    MATLAB source : graphics/specgraph/+matlab/+graphics/+chart/+internal/
        +saveload/createLinesForQuiverStruct.m, MATLAB R2025b installation
    Chebfun commit: 7574c77 (caller scope; geometry is MATLAB builtin source)
    Finite matching-shaped nonempty arrays only. Autoscale and xyzchk expansion
    belong to the caller. Head equality and eps perturbations are literal.
    """
    x, y, u, v = (jnp.asarray(a, dtype=jnp.float64).ravel(order="F")
                   for a in (x, y, u, v))
    if not (x.shape == y.shape == u.shape == v.shape) or x.size == 0:
        raise ValueError("expanded nonempty matching quiver arrays required")
    eps = jnp.finfo(jnp.float64).eps
    length_squared = u*u + v*v
    if z is not None:
        z, w = (jnp.asarray(a, dtype=jnp.float64).ravel(order="F") for a in (z, w))
        if z.shape != x.shape or w.shape != x.shape:
            raise ValueError("expanded matching 3D quiver arrays required")
        length_squared = length_squared + w*w
    norm = jnp.sqrt(length_squared)
    normxy = jnp.sqrt(u*u + v*v) + eps
    spanx = jnp.max(jnp.concatenate((x, x+u))) - jnp.min(jnp.concatenate((x, x+u)))
    spany = jnp.max(jnp.concatenate((y, y+v))) - jnp.min(jnp.concatenate((y, y+v)))
    span = jnp.maximum(spanx, spany)
    if z is not None:
        spanz = jnp.max(jnp.concatenate((z, z+w))) - jnp.min(jnp.concatenate((z, z+w)))
        span = jnp.maximum(span, spanz)
    cutoff = max_head_size * span
    beta = .25 * norm / normxy
    norm2 = jnp.where(norm < cutoff, 1., norm)
    norm2 = jnp.where(norm > cutoff, norm/cutoff, norm2)
    alpha = .33 / norm2
    base = jnp.stack((x, y), axis=-1)
    tip = jnp.stack((x+u, y+v), axis=-1)
    left = jnp.stack((x+u-alpha*(u+beta*(v+eps)),
                      y+v-alpha*(v-beta*(u+eps))), axis=-1)
    right = jnp.stack((x+u-alpha*(u-beta*(v+eps)),
                       y+v-alpha*(v+beta*(u+eps))), axis=-1)
    if z is not None:
        base = jnp.concatenate((base, z[:, None]), axis=1)
        tip = jnp.concatenate((tip, (z+w)[:, None]), axis=1)
        left = jnp.concatenate((left, (z+w-alpha*w)[:, None]), axis=1)
        right = jnp.concatenate((right, (z+w-alpha*w)[:, None]), axis=1)
    return jnp.stack((base, tip), axis=1), jnp.stack((left, tip, right), axis=1), base

