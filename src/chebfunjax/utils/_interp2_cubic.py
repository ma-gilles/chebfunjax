"""Uniform-grid cubic convolution used by the native Needle example.

Provenance
----------
Chebfun examples/opt/Needle.m (December 2013), interp2(..., 'cubic').
MATLAB R2025b interp2.m documents uniform-grid cubic convolution; captured
unit-grid basis values qualify Keys tension -1/2 and quadratic edge extension.
This adapter covers real increasing rectangular domains with at least four
samples per axis. It does not implement nonuniform spline fallback or the
other interp2 methods.
"""
import jax.numpy as jnp


def _weights(t):
    square, cube = t*t, t*t*t
    return jnp.stack((-.5*t+square-.5*cube,
                      1-2.5*square+1.5*cube,
                      .5*t+2*square-1.5*cube,
                      -.5*square+.5*cube), axis=-1)


def uniform_cubic_interp2(values, xq, yq, *, x_domain, y_domain):
    """Interpolate a uniform rows-y/columns-x grid at broadcast query points.

    ``x_domain`` and ``y_domain`` give the first and last sample coordinates.
    Outside queries return NaN, including nonfinite points. The function uses
    JAX throughout and supports JIT, vmap and differentiation through values.
    Domain ordering is a caller precondition; nonpositive spans return NaN.

    Provenance
    ----------
    MATLAB interp2 uniform cubic method; native Chebfun opt/Needle.m caller.
    Native basis fixture: tests/test_utils/interp2_cubic_native_r2025b.json.
    """
    values = jnp.asarray(values)
    if values.ndim != 2 or min(values.shape) < 4:
        raise ValueError("cubic interpolation requires a grid of at least 4 by 4")
    xd, yd = jnp.asarray(x_domain), jnp.asarray(y_domain)
    if xd.shape != (2,) or yd.shape != (2,):
        raise ValueError("each interpolation domain must contain two endpoints")
    xq, yq = jnp.broadcast_arrays(jnp.asarray(xq), jnp.asarray(yq))
    ny, nx = values.shape
    sx, sy = xd[1]-xd[0], yd[1]-yd[0]
    valid = (jnp.isfinite(xq) & jnp.isfinite(yq)
             & (xq >= xd[0]) & (xq <= xd[1])
             & (yq >= yd[0]) & (yq <= yd[1]) & (sx > 0) & (sy > 0))
    tx, ty = (xq-xd[0])/sx*(nx-1), (yq-yd[0])/sy*(ny-1)
    tx, ty = jnp.where(valid, tx, 0.), jnp.where(valid, ty, 0.)
    ix = jnp.clip(jnp.floor(tx).astype(jnp.int32), 0, nx-2)
    iy = jnp.clip(jnp.floor(ty).astype(jnp.int32), 0, ny-2)
    wx, wy = _weights(tx-ix), _weights(ty-iy)
    # Quadratic ghost samples preserve the native first/last-cell stencil.
    padded = jnp.concatenate(((3*values[:, 0]-3*values[:, 1]+values[:, 2])[:, None],
                              values,
                              (3*values[:, -1]-3*values[:, -2]+values[:, -3])[:, None]), axis=1)
    padded = jnp.concatenate(((3*padded[0]-3*padded[1]+padded[2])[None, :],
                              padded,
                              (3*padded[-1]-3*padded[-2]+padded[-3])[None, :]), axis=0)
    offsets = jnp.arange(4)
    stencil = padded[iy[..., None, None]+offsets[:, None],
                     ix[..., None, None]+offsets[None, :]]
    result = jnp.sum(wy[..., :, None]*stencil*wx[..., None, :], axis=(-2, -1))
    return jnp.where(valid, result, jnp.nan)
