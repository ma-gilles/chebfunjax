"""Conformal mapping to the unit disk.

Translated from MATLAB Chebfun (commit 7574c77): conformal.m.
Original: Copyright 2019 by L. N. Trefethen and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

References
----------
A. Gopal and L. N. Trefethen, "Representation of conformal maps by rational
functions", Numer. Math. 142 (2019), 359-382.

L. N. Trefethen, "Numerical conformal mapping with rational functions",
Comp. Meth. Funct. Th. 20 (2020), 369-387.
"""

from __future__ import annotations

import warnings
from typing import Callable, Tuple

import jax.numpy as jnp

from chebfunjax.utils.aaa import aaa

# ===========================================================================
# Public API
# ===========================================================================


def conformal(
    boundary_pts: jnp.ndarray,
    ctr: complex = 0.0,
    *,
    tol: float = 1e-5,
    method: str = "kerzman-stein",
) -> Tuple[Callable, Callable, jnp.ndarray, jnp.ndarray]:
    """Conformal map from a simply-connected region to the unit disk.

    [F, FINV, POL, POLINV] = CONFORMAL(C, CTR) computes a conformal map
    F of the region bounded by the complex curve C to the unit disk and its
    inverse FINV, with F(ctr) = 0 and F'(ctr) > 0.  Both maps are
    represented as barycentric rational functions via the AAA algorithm.

    Parameters
    ----------
    boundary_pts : Chebfun or complex array of shape (M,)
        Continuous boundary Chebfun, or periodic values on an equispaced grid.
        Arrays are a Python compatibility adapter to a periodic Chebfun.
        The boundary should be smooth (no corners).
    ctr : complex, default 0.0
        Interior point that maps to 0.  Must be inside the region.
    tol : float, default 1e-5
        Convergence tolerance (relative).
    method : {'kerzman-stein', 'poly'}, default 'kerzman-stein'
        Algorithm to use:
        - 'kerzman-stein': solve the Kerzman-Stein integral equation
          (Greenbaum-Caldwell), more robust for smooth domains.
        - 'poly': polynomial least-squares (faster for simple domains).

    Returns
    -------
    f : callable
        Forward conformal map.  ``f(z)`` maps region to unit disk.
        JIT-safe (it is the AAA rational approximant).
    finv : callable
        Inverse conformal map.  ``finv(w)`` maps unit disk to region.
        JIT-safe.
    pol : jnp.ndarray, complex
        Poles of the forward map.
    polinv : jnp.ndarray, complex
        Poles of the inverse map.

    Notes
    -----
    This is an experimental implementation suitable for smooth simple regions.
    Regions with corners or near-degeneracies may not converge.

    The Kerzman-Stein method sets up an O(M^2) linear system to find the
    boundary correspondence function.  The polynomial method is cheaper
    (O(M) iterations of a least-squares problem).

    Provenance
    ----------
    MATLAB source : conformal.m
    Chebfun commit: 7574c77
    Original authors: L. N. Trefethen, Anne Greenbaum, Trevor Caldwell.
        Copyright 2019 by The University of Oxford and The Chebfun Developers.

    Examples
    --------
    Ellipse centered at origin:

    >>> import jax.numpy as jnp
    >>> theta = jnp.linspace(0, 2*jnp.pi, 200, endpoint=False)
    >>> C = 2*jnp.cos(theta) + 1j*jnp.sin(theta)
    >>> f, finv, pol, polinv = conformal(C)

    See Also
    --------
    aaa
    """
    from chebfunjax.chebfun1d.chebfun import chebfun

    C = boundary_pts if hasattr(boundary_pts, "funs") else chebfun(
        jnp.asarray(boundary_pts, dtype=jnp.complex128).reshape(-1),
        domain=(0.0, 2 * jnp.pi), trig=True)
    winding = (C.diff() / (C - ctr)).sum() / (2j * jnp.pi)
    if bool(jnp.abs(winding - 1) > tol):
        raise ValueError("CONFORMAL:parseinputs: C must wind once counterclockwise around ctr")
    scl = (C - ctr).norm(jnp.inf)
    if method == "kerzman-stein":
        M, err = 300, float("inf")
        while err > tol:
            M += 300
            g, Z, W = _kerzman_stein((C - ctr) / scl, M)
            Z = Z * scl + ctr
            gc = g.trigcoeffs()
            err = float(jnp.linalg.norm(jnp.concatenate((gc[:10], gc[-10:]))))
            if err > tol and M >= 1200:
                warnings.warn("conformal did not converge", stacklevel=2)
                break
    elif method == "poly":
        W, Z = _poly_method(C, ctr, scl, tol=tol)
    else:
        raise ValueError("method must be 'kerzman-stein' or 'poly'")
    f0, pol, *_ = aaa(W, Z, tol=tol)
    zz = 1e-4 * scl * jnp.asarray([1, 1j, -1, -1j])
    dwdz = jnp.sum(f0(ctr + zz) / zz)
    rot = jnp.exp(-1j * jnp.angle(dwdz))
    f = lambda z: rot * f0(z)  # noqa: E731
    W = rot * W
    finv, polinv, *_ = aaa(Z, W, tol=tol)
    if pol.size and bool(jnp.any(_inside_polygon(pol, Z))):
        warnings.warn("conformal: pole of forward map inside region", stacklevel=2)
    if polinv.size and bool(jnp.min(jnp.abs(polinv)) < 1):
        warnings.warn("conformal: pole of inverse map inside unit disk", stacklevel=2)
    return f, finv, pol, polinv


def _inside_polygon(points, vertices):
    """JAX ray crossing for the source's discrete boundary pole check."""
    x, y = points.real[:, None], points.imag[:, None]
    a, b = vertices, jnp.roll(vertices, -1)
    dy = b.imag - a.imag
    crossing_x = (b.real - a.real) * (y - a.imag) / jnp.where(dy == 0, 1, dy) + a.real
    crossings = ((a.imag > y) != (b.imag > y)) & (x < crossing_x)
    return jnp.sum(crossings, axis=1) % 2 == 1


def _kerzman_stein(C, M):
    """Literal continuous arclength and KS stages, conformal.m 7574c77.

    Host decisions manage adaptive Chebfuns. Numerical arrays and linear
    algebra use JAX. No sampled arclength or interpolation surrogate is used.
    """
    from chebfunjax.chebfun1d.chebfun import chebfun

    S = C.arclength()
    s = abs(C.diff()).cumsum()
    dC_arg = C.diff().angle()
    _ = (dC_arg - dC_arg(0)).unwrap()
    u = s.inv()
    (_, lo), (_, hi) = u.minandmax()
    C = C.new_domain((lo, hi))
    D = C.compose_chebfun(u)
    Dprime = D.diff()
    ds = S / M
    svec = jnp.arange(M) * ds
    Z = D(svec)
    Dprimevec = Dprime(svec)
    tangent = Dprimevec / jnp.abs(Dprimevec)
    d = 1 / (2j * jnp.pi)
    gvec = d * jnp.conj(tangent / (0 - Z))
    delta = Z[:, None] - Z[None, :]
    diagonal = jnp.eye(M, dtype=bool)
    safe_delta = jnp.where(diagonal, 1, delta)
    kernel = jnp.conj(tangent[:, None] / safe_delta) + tangent[None, :] / (-safe_delta)
    A = jnp.eye(M, dtype=jnp.complex128) - jnp.where(diagonal, 0, d * kernel * ds)
    fvec = jnp.linalg.solve(A, gvec)
    Rprime = fvec ** 2
    W = -1j * tangent * (Rprime / jnp.abs(Rprime))
    g = chebfun(W, domain=(0, float(S)), trig=True)
    return g, Z, W


def _poly_method(C, ctr, scl, *, tol=1e-5):
    """Source resampling, Arnoldi and least squares; conformal.m 7574c77."""
    err, logn = float("inf"), 4.0
    a, b = float(C.domain.a), float(C.domain.b)
    while err > tol:
        n = int(2 ** logn + 0.5)
        M = 8 * n
        logn += 0.5
        Z = C(a + jnp.arange(1, M + 1) * (b - a) / M)
        Zscl = (Z - ctr) / scl
        G = -jnp.log(jnp.abs(Zscl))
        Q = jnp.ones((M, 1), dtype=jnp.complex128)
        for k in range(n):
            v = Zscl * Q[:, k]
            v = v - Q @ (Q.conj().T @ v) / M
            h = jnp.linalg.norm(v) / jnp.sqrt(M)
            Q = jnp.column_stack((Q, v / h))
        A = jnp.column_stack((Q.real, Q[:, 1:].imag))
        c = jnp.linalg.lstsq(A, G, rcond=None)[0]
        err = float(jnp.linalg.norm(A @ c - G, ord=jnp.inf))
        cc = c[:n + 1] - 1j * jnp.concatenate((jnp.zeros(1), c[n + 1:]))
        W = Zscl * jnp.exp(Q @ cc)
        if err > tol and logn >= 9.5:
            warnings.warn("conformal did not converge", stacklevel=2)
            break
    return W, Z
