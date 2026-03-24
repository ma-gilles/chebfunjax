"""ODE convenience wrappers for 1-D BVPs and IVPs.

Provides thin wrappers around :class:`~chebfunjax.operators.chebop.Chebop`
that mirror the MATLAB Chebfun ``ode45`` / ``ode113`` / ``bvp4c`` style API.
Users call :func:`ivp` or :func:`bvp` with the operator, domain, and boundary
or initial conditions, and receive a :class:`~chebfunjax.chebfun1d.Chebfun`.

Typical use::

    import jax.numpy as jnp
    from chebfunjax.chebfun1d.ode import bvp, ivp

    # BVP: u'' + u = 0,  u(0) = 0,  u(pi) = 0  =>  u = sin(x)
    u = bvp(lambda x, u: u.diff(2) + u, domain=(0.0, jnp.pi),
            lbc=0.0, rbc=0.0)

    # IVP: u' = u,  u(0) = 1  =>  u = exp(x)
    u = ivp(lambda x, u: u.diff() - u, domain=(0.0, 1.0), ic=[1.0])

Translated from MATLAB Chebfun @chebop usage patterns (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

Provenance
----------
MATLAB source : @chebop/chebop.m, @chebop/mldivide.m, examples/ode-eig/*.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

from typing import Callable

__all__ = ["bvp", "ivp", "eigs"]


def bvp(
    op: Callable,
    domain: tuple[float, float] = (-1.0, 1.0),
    lbc=None,
    rbc=None,
    f=0.0,
    n: int | None = None,
    n_min: int = 8,
    n_max: int = 2048,
    tol: float = 1e-10,
    max_iter: int = 15,
):
    """Solve a boundary-value problem.

    Constructs a :class:`~chebfunjax.operators.chebop.Chebop` and calls
    its :meth:`~chebfunjax.operators.chebop.Chebop.solve` method.

    Parameters
    ----------
    op : callable
        Differential operator.  Signature ``lambda x, u: ...`` or
        ``lambda u: ...``.  Must accept and return
        :class:`~chebfunjax.chebfun1d.Chebfun` objects.
    domain : (float, float), default (-1, 1)
        Physical domain.
    lbc : scalar, list, callable, or None
        Left boundary condition(s).  Passed directly to
        :attr:`~chebfunjax.operators.chebop.Chebop.lbc`.
    rbc : scalar, list, callable, or None
        Right boundary condition(s).
    f : scalar, callable, or Chebfun, default 0.0
        Right-hand side of the ODE.
    n : int or None
        Fixed discretization size (None = adaptive).
    n_min : int, default 8
        Minimum size for adaptive loop.
    n_max : int, default 2048
        Maximum size for adaptive loop.
    tol : float, default 1e-10
        Convergence tolerance.
    max_iter : int, default 15
        Maximum Newton iterations (nonlinear problems only).

    Returns
    -------
    u : :class:`~chebfunjax.chebfun1d.Chebfun`
        Solution satisfying the ODE and boundary conditions.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from chebfunjax.chebfun1d.ode import bvp
    >>> # u'' = -1, u(-1) = 0, u(1) = 0  =>  u = (1 - x^2) / 2
    >>> u = bvp(lambda x, u: u.diff(2), domain=(-1.0, 1.0),
    ...         lbc=0.0, rbc=0.0, f=-1.0)
    >>> abs(float(u(jnp.float64(0.0))) - 0.5) < 1e-10
    True

    Provenance
    ----------
    MATLAB source : @chebop/mldivide.m, @chebop/solvebvpLinear.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.operators.chebop import Chebop

    N = Chebop(op, domain=domain, lbc=lbc, rbc=rbc)
    return N.solve(
        f,
        n=n,
        n_min=n_min,
        n_max=n_max,
        tol=tol,
        max_iter=max_iter,
    )


def ivp(
    op: Callable,
    domain: tuple[float, float] = (-1.0, 1.0),
    ic=None,
    f=0.0,
    n: int | None = None,
    n_min: int = 8,
    n_max: int = 2048,
    tol: float = 1e-10,
    max_iter: int = 15,
):
    """Solve an initial-value problem.

    An IVP is a BVP where all boundary conditions are imposed at the *left*
    endpoint.  This wrapper passes ``ic`` as ``lbc`` with ``rbc=None``.

    Parameters
    ----------
    op : callable
        Differential operator.  Signature ``lambda x, u: ...`` or
        ``lambda u: ...``.
    domain : (float, float), default (-1, 1)
        Physical domain (left endpoint = initial time).
    ic : scalar or list of scalars
        Initial condition(s).  If a scalar, ``u(a) = ic``.  If a list
        ``[u0, u1, ...]``, then ``u(a) = u0``, ``u'(a) = u1``, etc.
    f : scalar, callable, or Chebfun, default 0.0
        Right-hand side.
    n : int or None
        Fixed discretization size (None = adaptive).
    n_min : int, default 8
        Minimum adaptive size.
    n_max : int, default 2048
        Maximum adaptive size.
    tol : float, default 1e-10
        Convergence tolerance.
    max_iter : int, default 15
        Maximum Newton iterations.

    Returns
    -------
    u : :class:`~chebfunjax.chebfun1d.Chebfun`
        Solution.

    Examples
    --------
    >>> from chebfunjax.chebfun1d.ode import ivp
    >>> import jax.numpy as jnp
    >>> # u' = u,  u(0) = 1  =>  u = exp(x)
    >>> u = ivp(lambda x, u: u.diff() - u, domain=(0.0, 1.0), ic=[1.0])
    >>> abs(float(u(1.0)) - float(jnp.exp(jnp.float64(1.0)))) < 1e-8
    True

    Provenance
    ----------
    MATLAB source : @chebop/mldivide.m (IVP case: all BCs at left endpoint)
    Chebfun commit: 7574c77
    """
    return bvp(
        op,
        domain=domain,
        lbc=ic,
        rbc=None,
        f=f,
        n=n,
        n_min=n_min,
        n_max=n_max,
        tol=tol,
        max_iter=max_iter,
    )


def eigs(
    op: Callable,
    domain: tuple[float, float] = (-1.0, 1.0),
    lbc=None,
    rbc=None,
    k: int = 6,
    n: int | None = None,
    n_default: int = 64,
    sigma=None,
):
    """Compute eigenvalues of a linear differential operator.

    Constructs a :class:`~chebfunjax.operators.chebop.Chebop` and calls
    its :meth:`~chebfunjax.operators.chebop.Chebop.eigs` method.

    Parameters
    ----------
    op : callable
        Linear differential operator.
    domain : (float, float), default (-1, 1)
        Physical domain.
    lbc : scalar, list, callable, or None
        Left boundary condition(s).
    rbc : scalar, list, callable, or None
        Right boundary condition(s).
    k : int, default 6
        Number of eigenvalues.
    n : int or None
        Discretization size (None = ``n_default``).
    n_default : int, default 64
        Default size when ``n`` is None.
    sigma : scalar, str, or None
        Target eigenvalue or selector string.

    Returns
    -------
    lam : jnp.ndarray, shape (k,)
        Selected eigenvalues.

    Examples
    --------
    >>> from chebfunjax.chebfun1d.ode import eigs
    >>> # -u'' with Dirichlet BCs on [-1,1]: eigenvalues (k*pi/2)^2
    >>> lam = eigs(lambda x, u: -u.diff(2), domain=(-1.0, 1.0),
    ...            lbc=0.0, rbc=0.0, k=4)
    >>> lam  # doctest: +SKIP
    Array([2.46..., 9.86..., 22.2..., 39.4...], dtype=float64)

    Provenance
    ----------
    MATLAB source : @chebop/eigs.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.operators.chebop import Chebop

    N = Chebop(op, domain=domain, lbc=lbc, rbc=rbc)
    return N.eigs(k=k, n=n, n_default=n_default, sigma=sigma)
