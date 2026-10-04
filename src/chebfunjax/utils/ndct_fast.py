"""JAX implementation of Chebfun @chebfun/ndct.m fast evaluator.

This is the K=16 low-rank nonuniform DCT/FFT algorithm from Chebfun
commit 7574c77. It evaluates Chebyshev coefficient columns at supplied
radian angles. The MATLAB wrapper chebcoeffs2legvals uses theta=acos(x)
at Legendre nodes. The source final pure-imaginary coercion is deliberately
not applied here; callers can decide whether that wrapper behavior is wanted.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Exact decimal table from besselCoeffs(K) in @chebfun/ndct.m (K=16).
_BESSEL_COEFFS = jnp.asarray([
    [0.725276916440514, 0.0, -0.263810811846140, 0.0, 0.010721845410224, 0.0, -0.000188568764214, 0.0, 0.000001845983729, 0.0, -0.000000011505371, 0.0, 0.000000000049650, 0.0, -0.000000000000157, 0.0],
    [0.0, -1.237209415222620j, 0.0, 0.106368016662154j, 0.0, -0.002843803888523j, 0.0, 0.000037314601460j, 0.0, -0.000000291470262j, 0.0, 0.000000001511615j, 0.0, -0.000000000005586j, 0.0, 0.000000000000015j],
    [-0.263810811846140, 0.0, -0.249420239398158, 0.0, 0.014106236744931, 0.0, -0.000281370589589, 0.0, 0.000002945880969, 0.0, -0.000000019147182, 0.0, 0.000000000085036, 0.0, -0.000000000000275, 0.0],
    [0.0, 0.106368016662154j, 0.0, 0.033077433013562j, 0.0, -0.001395694044102j, 0.0, 0.000022213402600j, 0.0, -0.000000193519978j, 0.0, 0.000000001077127j, 0.0, -0.000000000004183j, 0.0, 0.000000000000012j],
    [0.010721845410224, 0.0, 0.014106236744931, 0.0, 0.003272735109015, 0.0, -0.000110186049486, 0.0, 0.000001459236547, 0.0, -0.000000010886489, 0.0, 0.000000000052987, 0.0, -0.000000000000183, 0.0],
    [0.0, -0.002843803888523j, 0.0, -0.001395694044102j, 0.0, -0.000258373068366j, 0.0, 0.000007238310730j, 0.0, -0.000000082089523j, 0.0, 0.000000000535538j, 0.0, -0.000000000002316j, 0.0, 0.000000000000007j],
    [-1.885687642135953e-04, 0.0, -2.813705895892196e-04, 0.0, -1.101860494855995e-04, 0.0, -1.697297037000288e-05, 0.0, 4.071920170840575e-07, 0.0, -4.038224124439603e-09, 0.0, 2.340743035825754e-11, 0.0, -9.107573296934322e-14, 0.0],
    [0.0, 3.731460145969448e-05j, 0.0, 2.221340260014939e-05j, 0.0, 7.238310730031504e-06j, 0.0, 9.548164341984993e-07j, 0.0, -2.003096829508016e-08j, 0.0, 1.765036265437698e-10j, 0.0, -9.204996298527833e-13j, 0.0, 3.255202808838026e-15j],
    [1.845983728936490e-06, 0.0, 2.945880968932915e-06, 0.0, 1.459236547111028e-06, 0.0, 4.071920170840575e-07, 0.0, 4.697021778082508e-08, 0.0, -8.755181580605211e-10, 0.0, 6.941023444886516e-12, 0.0, -3.290023459530936e-14, 0.0],
    [0.0, -2.914702622193488e-07j, 0.0, -1.935199775867226e-07j, 0.0, -8.208952270529682e-08j, 0.0, -2.003096829508016e-08j, 0.0, -2.052985055408922e-09j, 0.0, 3.442984249400081e-11j, 0.0, -2.480840754980324e-13j, 0.0, 1.077723939077946e-15j],
    [-1.150537142155097e-08, 0.0, -1.914718184703716e-08, 0.0, -1.088648898321836e-08, 0.0, -4.038224124439603e-09, 0.0, -8.755181580605211e-10, 0.0, -8.073385051984342e-11, 0.0, 1.230581586777325e-12, 0.0, -8.126572662991549e-15, 0.0],
    [0.0, 1.511615196341062e-09j, 0.0, 1.077126955246971e-09j, 0.0, 5.355382878799667e-10j, 0.0, 1.765036265437698e-10j, 0.0, 3.442984249400081e-11j, 0.0, 2.885566203117638e-12j, 0.0, -4.031057077166038e-14j, 0.0, 2.456927663965262e-16j],
    [4.965029850164547e-11, 0.0, 8.503608974664060e-11, 0.0, 5.298703065162090e-11, 0.0, 2.340743035825754e-11, 0.0, 6.941023444886516e-12, 0.0, 1.230581586777325e-12, 0.0, 9.452345289165531e-14, 0.0, -1.218719878432285e-15, 0.0],
    [0.0, -5.586166703739336e-12j, 0.0, -4.183174389936338e-12j, 0.0, -2.315969292837316e-12j, 0.0, -9.204996298527833e-13j, 0.0, -2.480840754980324e-13j, 0.0, -4.031057077166038e-14j, 0.0, 2.857751919953105e-15j, 0.0, 0.0],
    [-1.571252307825089e-13, 0.0, -2.747999062823869e-13, 0.0, -1.828391460048661e-13, 0.0, -9.107573296934322e-14, 0.0, -3.290023459530936e-14, 0.0, -8.126572662991549e-15, 0.0, -1.218719878432285e-15, 0.0, 0.0, 0.0],
    [0.0, 1.545890088268538e-14j, 0.0, 1.201101735269839e-14j, 0.0, 7.190168405679147e-15j, 0.0, 3.255202808838026e-15j, 0.0, 1.077723939077946e-15j, 0.0, 2.456927663965262e-16j, 0.0, 0.0, 0.0, 0.0],
], dtype=jnp.complex128)


def _cheb_polynomials(x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate T_0,...,T_15 by the source three-term recurrence."""
    values = [jnp.ones_like(x), x]
    for _ in range(2, 16):
        values.append(2 * x * values[-1] - values[-2])
    return jnp.stack(values, axis=-1)


@jax.jit
def _ndct_fast(coeffs: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Evaluate Chebyshev coefficients at arbitrary angles using source K=16 NDCT.

    Args:
        coeffs: Shape ``(n,)`` or ``(n, m)`` coefficient vector/columns.
        theta: One-dimensional vector of evaluation angles in radians.

    Returns:
        Shape ``(len(theta),)`` for vector input or ``(len(theta), m)`` for
        matrix input. Real coefficient inputs produce real outputs. Complex
        inputs preserve the full complex result. MATLAB ndct.m additionally
        coerces purely imaginary coefficient inputs to ``imag(vals)``; that
        wrapper quirk is intentionally left to the caller.

    Provenance:
        MATLAB source: @chebfun/ndct.m, besselCoeffs and ChebP helpers.
        Chebfun commit: 7574c77.
        Algorithm: Ruiz-Antolin and Townsend low-rank NUFFT, K=16.
    """
    coeffs = jnp.asarray(coeffs)
    coeffs = coeffs.astype(jnp.result_type(coeffs.dtype, jnp.float64))
    theta = jnp.real(jnp.asarray(theta)).astype(jnp.float64)
    vector_input = coeffs.ndim == 1
    if vector_input:
        coeffs = coeffs[:, None]
    elif coeffs.ndim != 2:
        raise ValueError("coeffs must be a vector or a 2-D column matrix")
    if theta.ndim != 1:
        raise ValueError("theta must be a one-dimensional angle vector")

    n, ncols = coeffs.shape
    ntheta = theta.shape[0]
    if n == 0:
        values = jnp.zeros((ntheta, ncols), dtype=coeffs.dtype)
        return values[:, 0] if vector_input else values
    if n == 1:
        values = jnp.broadcast_to(coeffs[0, :], (ntheta, ncols))
        return values[:, 0] if vector_input else values

    # MATLAB mirrors the coefficients to convert the Chebyshev sum to a DFT.
    # Only nonconstant coefficients are halved before mirroring.
    c = coeffs.at[1:, :].divide(2)
    c = jnp.concatenate((c[:0:-1, :], c), axis=0)
    N = 2 * n - 1

    # MATLAB round(): nearest integer, ties away from zero.
    scaled_theta = N * (theta / (2 * jnp.pi))
    s = jnp.where(scaled_theta >= 0,
                  jnp.floor(scaled_theta + 0.5),
                  jnp.ceil(scaled_theta - 0.5)).astype(jnp.int32)
    t = jnp.mod(s, N)
    ds = 2 * (scaled_theta - s)

    U = _cheb_polynomials(ds) @ _BESSEL_COEFFS
    k = 2 * jnp.arange(-(n - 1), n, dtype=jnp.float64) / N
    V = _cheb_polynomials(k)

    tmp = c[:, :, None] * V[:, None, :]
    tmp = jnp.fft.ifftshift(tmp, axes=0)
    transformed = jnp.fft.fft(tmp, axis=0)
    values = jnp.sum(U[:, None, :] * transformed[t, :, :], axis=2)
    if not jnp.issubdtype(coeffs.dtype, jnp.complexfloating):
        values = jnp.real(values)
    return values[:, 0] if vector_input else values
