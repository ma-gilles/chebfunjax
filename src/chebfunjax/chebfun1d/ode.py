"""ODE convenience wrappers for 1-D BVPs and IVPs.

Provides thin wrappers around :class:`~chebfunjax.operators.chebop.Chebop`
that mirror the MATLAB Chebfun ``ode45`` / ``ode113`` / ``bvp4c`` / ``bvp5c``
style API.  Users call :func:`ivp`, :func:`bvp`, :func:`bvp4c`, or
:func:`bvp5c` with the operator, domain, and boundary conditions, and
receive a :class:`~chebfunjax.chebfun1d.Chebfun`.

Typical use::

    import jax.numpy as jnp
    from chebfunjax.chebfun1d.ode import bvp, bvp4c, ivp

    # BVP: u'' + u = 0,  u(0) = 0,  u(pi) = 0  =>  u = sin(x)
    u = bvp(lambda x, u: u.diff(2) + u, domain=(0.0, jnp.pi),
            lbc=0.0, rbc=0.0)

    # BVP4C style (mirrors MATLAB chebfun.bvp4c):
    u = bvp4c(lambda x, u: u.diff(2) + u, domain=(0.0, jnp.pi),
              lbc=0.0, rbc=0.0)

    # IVP: u' = u,  u(0) = 1  =>  u = exp(x)
    u = ivp(lambda x, u: u.diff() - u, domain=(0.0, 1.0), ic=[1.0])

Translated from MATLAB Chebfun @chebop and @chebfun/bvp4c usage patterns
(commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

Provenance
----------
MATLAB source : @chebop/chebop.m, @chebop/mldivide.m,
                @chebfun/bvp4c.m, @chebfun/bvp5c.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

from typing import Callable

__all__ = ["bvp", "bvp4c", "bvp5c", "ivp", "eigs"]


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


# ============================================================================
# V05: bvp4c / bvp5c — collocation BVP solvers  (MATLAB chebfun.bvp4c/bvp5c)
# ============================================================================


def bvp4c(
    op: Callable,
    domain: tuple[float, float] = (-1.0, 1.0),
    lbc=None,
    rbc=None,
    f=0.0,
    n: int | None = None,
    n_min: int = 8,
    n_max: int = 2048,
    tol: float = 1e-6,
    max_iter: int = 15,
    refine: bool = True,
    n_refine: int = 2,
):
    """Solve a BVP using 4th-order collocation (MATLAB ``bvp4c`` analogue).

    Mirrors MATLAB Chebfun's ``chebfun.bvp4c`` interface: wraps
    :func:`bvp` with collocation refinement enabled by default and a
    default tolerance of ``1e-6`` to match MATLAB's ``bvp4c`` defaults.

    The collocation refinement loop doubles the discretization size up to
    ``n_refine`` times if the initial solve does not meet ``tol``.

    Parameters
    ----------
    op : callable
        Differential operator.  Signature ``lambda x, u: ...`` or
        ``lambda u: ...``.
    domain : (float, float), default (-1, 1)
        Physical domain.
    lbc : scalar, list, callable, or None
        Left boundary condition(s).
    rbc : scalar, list, callable, or None
        Right boundary condition(s).
    f : scalar, callable, or Chebfun, default 0.0
        Right-hand side.
    n : int or None
        Fixed starting discretization size (None = use ``n_min``).
    n_min : int, default 8
        Minimum adaptive size.
    n_max : int, default 2048
        Maximum adaptive size.
    tol : float, default 1e-6
        Convergence tolerance (matches MATLAB bvp4c default RELTOL).
    max_iter : int, default 15
        Maximum Newton iterations per refinement level.
    refine : bool, default True
        If ``True``, attempt collocation refinement (doubling ``n``) when
        the initial solve is not sufficiently resolved.
    n_refine : int, default 2
        Maximum number of refinement doublings.

    Returns
    -------
    u : :class:`~chebfunjax.chebfun1d.Chebfun`
        Solution satisfying the ODE and boundary conditions.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from chebfunjax.chebfun1d.ode import bvp4c
    >>> # u'' = -1, u(-1) = 0, u(1) = 0  =>  u = (1 - x^2) / 2
    >>> u = bvp4c(lambda x, u: u.diff(2), domain=(-1.0, 1.0),
    ...           lbc=0.0, rbc=0.0, f=-1.0)
    >>> abs(float(u(jnp.float64(0.0))) - 0.5) < 1e-6
    True

    Notes
    -----
    MATLAB's ``bvp4c`` uses a 4th-order finite-difference collocation method
    on a non-uniform mesh that is adaptively refined.  Our implementation
    uses Chebyshev spectral collocation via :class:`~chebfunjax.operators.Chebop`,
    which is typically much more accurate.  The "collocation refinement"
    here refers to increasing the Chebyshev degree, not the mesh density.

    Provenance
    ----------
    MATLAB source : @chebfun/bvp4c.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    bvp5c, bvp, ivp
    """
    return _bvp_colloc(
        op=op, domain=domain, lbc=lbc, rbc=rbc, f=f,
        n=n, n_min=n_min, n_max=n_max, tol=tol,
        max_iter=max_iter, refine=refine, n_refine=n_refine,
        label="bvp4c",
    )


def bvp5c(
    op: Callable,
    domain: tuple[float, float] = (-1.0, 1.0),
    lbc=None,
    rbc=None,
    f=0.0,
    n: int | None = None,
    n_min: int = 8,
    n_max: int = 2048,
    tol: float = 1e-6,
    max_iter: int = 15,
    refine: bool = True,
    n_refine: int = 3,
):
    """Solve a BVP using 5th-order collocation (MATLAB ``bvp5c`` analogue).

    Mirrors MATLAB Chebfun's ``chebfun.bvp5c`` interface: wraps
    :func:`bvp` with collocation refinement enabled.  The default
    ``n_refine=3`` allows more refinement doublings than :func:`bvp4c`,
    matching the higher accuracy expected from a 5th-order method.

    Parameters
    ----------
    op : callable
        Differential operator.
    domain : (float, float), default (-1, 1)
        Physical domain.
    lbc : scalar, list, callable, or None
        Left boundary condition(s).
    rbc : scalar, list, callable, or None
        Right boundary condition(s).
    f : scalar, callable, or Chebfun, default 0.0
        Right-hand side.
    n : int or None
        Fixed starting size (None = use ``n_min``).
    n_min : int, default 8
        Minimum adaptive size.
    n_max : int, default 2048
        Maximum adaptive size.
    tol : float, default 1e-6
        Convergence tolerance.
    max_iter : int, default 15
        Maximum Newton iterations per refinement level.
    refine : bool, default True
        Enable collocation refinement.
    n_refine : int, default 3
        Maximum number of refinement doublings.

    Returns
    -------
    u : :class:`~chebfunjax.chebfun1d.Chebfun`
        Solution satisfying the ODE and boundary conditions.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from chebfunjax.chebfun1d.ode import bvp5c
    >>> # u'' = -1, u(-1) = 0, u(1) = 0  =>  u = (1 - x^2) / 2
    >>> u = bvp5c(lambda x, u: u.diff(2), domain=(-1.0, 1.0),
    ...           lbc=0.0, rbc=0.0, f=-1.0)
    >>> abs(float(u(jnp.float64(0.0))) - 0.5) < 1e-6
    True

    Provenance
    ----------
    MATLAB source : @chebfun/bvp5c.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    bvp4c, bvp, ivp
    """
    return _bvp_colloc(
        op=op, domain=domain, lbc=lbc, rbc=rbc, f=f,
        n=n, n_min=n_min, n_max=n_max, tol=tol,
        max_iter=max_iter, refine=refine, n_refine=n_refine,
        label="bvp5c",
    )


# ---------------------------------------------------------------------------
# Shared implementation for bvp4c / bvp5c
# ---------------------------------------------------------------------------


def _bvp_colloc(
    *,
    op: Callable,
    domain: tuple[float, float],
    lbc,
    rbc,
    f,
    n: int | None,
    n_min: int,
    n_max: int,
    tol: float,
    max_iter: int,
    refine: bool,
    n_refine: int,
    label: str,
):
    """Collocation BVP solve with optional refinement.

    Delegates to :func:`bvp`.  If ``refine=True``, re-solves with doubled
    ``n_min`` if the solution's tail coefficients have not decayed to ``tol``.

    Provenance
    ----------
    MATLAB source : @chebfun/bvp4c.m, @chebfun/bvp5c.m
    Chebfun commit: 7574c77
    """
    import jax.numpy as jnp

    # Initial solve
    u = bvp(
        op=op, domain=domain, lbc=lbc, rbc=rbc, f=f,
        n=n, n_min=n_min, n_max=n_max, tol=tol, max_iter=max_iter,
    )

    if not refine:
        return u

    # Check tail-coefficient decay; refine if not yet converged
    for _ in range(n_refine):
        try:
            piece = u.funs[0]
            c = piece.coeffs  # Chebyshev coefficients
            tail_abs = float(jnp.max(jnp.abs(c[-max(1, len(c) // 4):])))
            if tail_abs < tol:
                break  # already converged
            # Try again with double the size
            n_new = min(n_max, (piece.n or n_min) * 2)
            u = bvp(
                op=op, domain=domain, lbc=lbc, rbc=rbc, f=f,
                n=n_new, n_min=n_min, n_max=n_max,
                tol=tol, max_iter=max_iter,
            )
        except Exception:
            break  # If refinement fails, return best result so far

    return u
