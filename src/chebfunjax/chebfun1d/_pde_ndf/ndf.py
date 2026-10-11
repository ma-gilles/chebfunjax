"""Real64 NDF startup for dense constant-mass DAE and identity-mass ODE.

Source: MATLAB R2025b ode15s.m (Copyright 1984-2024 The MathWorks, Inc.).
Constant full mass, default order5/NDF, componentwise error control. JAX array
operations are dispatched in source statement order by this host controller;
outer-jitting this function is not qualified and could contract statements.

Provenance
----------
MATLAB R2025b ode15s.m startup and constant-mass initialization.
Chebfun native commit: 7574c77680d7e82b79626300bf255498271a72df.
Native ODE runtime/source release: MATLAB R2025b.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl

EPS = 2.220446049250313e-16


def positive_max(a, b):
    # Nonnegative finite IEEE ordering preserves subnormal values on FTZ CPUs.
    aa = jax.lax.bitcast_convert_type(jnp.asarray(a, jnp.float64), jnp.uint64)
    bb = jax.lax.bitcast_convert_type(jnp.asarray(b, jnp.float64), jnp.uint64)
    return jax.lax.bitcast_convert_type(jnp.maximum(aa, bb), jnp.float64)


def sixteen_eps(t):
    """Exact binary64 16*eps(t), including t=0 -> word0x10, without FTZ."""
    raw = jax.lax.bitcast_convert_type(jnp.asarray(t, jnp.float64), jnp.uint64)
    exponent = (raw >> jnp.uint64(52)) & jnp.uint64(2047)
    # For exponent E>=49 the result is normal with exponent E-48.
    normal = (jnp.maximum(exponent, jnp.uint64(49)) - jnp.uint64(48)) << jnp.uint64(52)
    shift = jnp.minimum(jnp.maximum(exponent, jnp.uint64(1)) + jnp.uint64(3), jnp.uint64(51))
    subnormal = jnp.uint64(1) << shift
    word = jnp.where(exponent >= 49, normal, subnormal)
    return jax.lax.bitcast_convert_type(word, jnp.float64)


def startup(
    y,
    yp,
    jac,
    mass,
    *,
    t=0.0,
    rtol=1e-6,
    threshold=1.0,
    userhmin=0.0,
    userhmax=0.1,
    htspan=0.1,
    dae=True,
    rhs=None,
):
    tiny = sixteen_eps(t)
    hmin = positive_max(tiny, userhmin)
    hmax = positive_max(tiny, userhmax)
    wt = jnp.maximum(jnp.abs(y), threshold)
    rh = 1.25 * jnp.max(jnp.abs(yp / wt))
    rh = rh / jnp.sqrt(jnp.asarray(rtol))
    absh = jnp.minimum(hmax, htspan)
    if bool(absh * rh > 1):
        absh = 1 / rh
    absh = positive_max(absh, hmin)
    if not dae:
        if rhs is None:
            raise ValueError("Ordinary ODE startup requires its time-dependent RHS")
        # ode15s non-DAE second-derivative estimate, Mtype=0 (identity mass).
        tdel = (
            t + jnp.minimum(jnp.sqrt(EPS) * jnp.maximum(jnp.abs(t), jnp.abs(t + absh)), absh)
        ) - t
        f0 = rhs(t, y)
        f1 = rhs(t + tdel, y)
        derivative = (f1 - f0) / tdel + jac @ yp
        rh = 1.25 * jnp.sqrt(0.5 * jnp.max(jnp.abs(derivative / wt)) / rtol)
        absh = jnp.minimum(hmax, htspan)
        if bool(absh * rh > 1):
            absh = 1 / rh
        absh = positive_max(absh, hmin)
    h = absh
    G = jnp.asarray([1.0, 3 / 2, 11 / 6, 25 / 12, 137 / 60])
    alpha = jnp.asarray([-37 / 200, -1 / 9, -0.0823, -0.0415, 0.0])
    invGa = 1 / (G * (1 - alpha))
    erconst = alpha * G + 1 / jnp.arange(2, 7, dtype=jnp.float64)
    dif = jnp.zeros((y.size, 7), dtype=y.dtype).at[:, 0].set(h * yp)
    hinvGak = h * invGa[0]
    product = hinvGak * jac
    matrix = mass - product
    rowscale = 1 / jnp.max(jnp.abs(matrix), axis=1) if dae else jnp.ones_like(y)
    scaled = rowscale[:, None] * matrix
    factor = jl.lu_factor(scaled)
    return dict(
        dae=dae,
        userhmin=userhmin,
        userhmax=userhmax,
        y=y,
        yp=yp,
        jac=jac,
        mass=mass,
        t=jnp.asarray(t),
        rtol=jnp.asarray(rtol),
        threshold=jnp.asarray(threshold),
        hmin=hmin,
        hmax=hmax,
        rh=rh,
        h=h,
        absh=absh,
        dif=dif,
        G=G,
        invGa=invGa,
        erconst=erconst,
        hinvGak=hinvGak,
        RowScale=rowscale,
        Miter=scaled,
        Factors=factor[0],
        piv=factor[1],
    )
