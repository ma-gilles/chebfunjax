"""Source coefficient/grid data for the five default Ballfun plot surfaces.

This module supplies numerical surface data, not interpolated lighting or a
MATLAB-compatible graphics backend. All transforms and geometry use JAX.

Provenance
----------
MATLAB source: @ballfun/plot.m47–108, @ballfun/coeffs3.m,
    @ballfun/coeffs2vals.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Original authors: Copyright 2019 University of Oxford and Chebfun Developers.
"""

import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import Trigtech
from chebfunjax.utils.quadrature import chebpts, trigpts


@dataclass(frozen=True)
class BallPlotSurface:
    """One source slice expressed in Cartesian coordinates and source CData."""

    kind: str
    index: int
    x: jax.Array
    y: jax.Array
    z: jax.Array
    values: jax.Array


@dataclass(frozen=True)
class BallPlotData:
    """Five surfaces plus the source grids and retained cylindrical values."""

    shape: tuple[int, int, int]
    radius: jax.Array
    elevation: jax.Array
    longitude: jax.Array
    values: jax.Array
    radius_index: int
    surfaces: tuple[BallPlotSurface, ...]


def plot_grid_shape(shape):
    """Apply native minimum dimensions and radial/angular congruences."""
    m, n, p = (int(v) for v in shape)
    m, n, p = max(25, m), max(28, n), max(28, p)
    return m + (1 - m % 6) % 6, n + (-n) % 4, p + (-p) % 4


def prolong_plot_coefficients(coefficients, shape):
    """Batch native lambda, radial, theta aliases in their original order.

    Only prolongation is required by plotBall's grid rule. The aliases retain
    even-length input Nyquist splitting; no analytic coefficient is discarded.
    """
    c = jnp.asarray(coefficients, dtype=jnp.complex128)
    m0, n0, p0 = c.shape
    m, n, p = shape
    if m < m0 or n < n0 or p < p0:
        raise ValueError("Ballfun plot coefficient dimensions must not shrink")
    c = Trigtech.alias(c.transpose(1, 0, 2).reshape(n0, m0 * p0), n)
    c = c.reshape(n, m0, p0).transpose(1, 0, 2)
    c = Chebtech2.alias(c.reshape(m0, n * p0), m).reshape(m, n, p0)
    c = Trigtech.alias(c.transpose(2, 1, 0).reshape(p0, n * m), p)
    return c.reshape(p, n, m).transpose(2, 1, 0)


def source_coefficients_to_values(coefficients):
    """Native radial-first CFF inverse transform, including complex roundoff."""
    c = jnp.asarray(coefficients, dtype=jnp.complex128)
    m, n, p = c.shape
    if m > 1:
        c = c.at[1:m - 1].divide(2)
        c = jnp.fft.fft(jnp.concatenate((c, c[m - 2:0:-1])), axis=0)
        c = c[m - 1::-1]
    kn = jnp.arange(n) - n // 2
    kp = jnp.arange(p) - p // 2
    sign_n = jnp.where(kn % 2 == 0, 1., -1.)
    scale_p = (n * p) * jnp.where(kp % 2 == 0, 1., -1.)
    factors = (sign_n[:, None] * scale_p[None, :])[None, :, :]
    c = jnp.fft.ifftshift(jnp.fft.ifftshift(c * factors, axes=1), axes=2)
    return jnp.fft.ifft(jnp.fft.ifft(c, axis=1), axis=2)


def source_slice_data(values):
    """Apply native real projection, theta ordering and longitude closure.

    Returned surface array orientations are r-by-longitude, elevation-by-
    longitude, and r-by-elevation respectively. These represent the native
    slice point sets; MATLAB slice handle-array orientation is not claimed.
    """
    m, n, p = values.shape
    if plot_grid_shape((m, n, p)) != (m, n, p):
        raise ValueError("Ballfun plot values require native plot dimensions")
    r = chebpts(m)[m // 2:]
    lam = jnp.concatenate((jnp.pi * trigpts(n)[0], jnp.asarray([jnp.pi])))
    el = (jnp.concatenate((jnp.pi * trigpts(p)[0], jnp.asarray([jnp.pi])))
          - jnp.pi / 2)[p // 2:]
    theta_order = jnp.concatenate((jnp.asarray([0]), jnp.arange(p - 1, p // 2 - 1, -1)))
    retained = jnp.real(values).transpose(0, 2, 1)[m // 2:, theta_order, :]
    retained = jnp.concatenate((retained, retained[:, :, :1]), axis=2)
    radius_index = int(jnp.argmin(jnp.abs(r - .5)))
    surfaces = []

    def append(kind, index, rr, tt, ll, cdata):
        x, y, z = jnp.broadcast_arrays(rr * jnp.cos(tt) * jnp.cos(ll),
                                      rr * jnp.cos(tt) * jnp.sin(ll),
                                      rr * jnp.sin(tt))
        surfaces.append(BallPlotSurface(kind, index, x, y, z, cdata))

    rr, ll = jnp.meshgrid(r, lam, indexing="ij")
    for index in (0, p // 4):
        append("elevation", index, rr, el[index], ll, retained[:, index, :])
    tt, ll = jnp.meshgrid(el, lam, indexing="ij")
    append("radius", radius_index, r[radius_index], tt, ll, retained[radius_index])
    rr, tt = jnp.meshgrid(r, el, indexing="ij")
    for index in (0, n // 4):
        append("longitude", index, rr, tt, lam[index], retained[:, :, index])
    return BallPlotData((m, n, p), r, el, lam, retained, radius_index, tuple(surfaces))


def ball_plot_data(ball):
    """Build literal default-plot data without evaluating physical grid points."""
    if ball.isempty():
        raise ValueError("CHEBFUN:BALLFUN:plot:isempty: Function is empty.")
    if not ball.is_real:
        warnings.warn("CHEBFUN:BALLFUN:plot:isReal: Function is not real, plotting the real part.",
                      UserWarning, stacklevel=2)
    shape = plot_grid_shape(ball.shape)
    coefficients = prolong_plot_coefficients(ball.coeffs, shape)
    return source_slice_data(source_coefficients_to_values(coefficients))
