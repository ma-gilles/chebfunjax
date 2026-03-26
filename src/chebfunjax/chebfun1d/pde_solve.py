# uses-numpy: ODE integration via scipy (not JIT-safe by design)
"""Full method-of-lines PDE solver — ``pdeSolve``.

Implements a method-of-lines (MOL) PDE solver that discretises the spatial
variable on a Chebyshev collocation grid and integrates the resulting large
ODE system in time with ``scipy.integrate.solve_ivp``.  The default
time-integration method is ``"BDF"`` (backward differentiation formulae),
matching the spirit of MATLAB's ``pdeSolve``/``pde15s`` which use ``ode15s``.

Supported boundary conditions
------------------------------
- **Dirichlet** (scalar ``lbc`` / ``rbc``): value ``u(a) = lbc``.
- **Neumann** (dict ``{"neumann": value}``): derivative ``u'(a) = lbc``.
- **Periodic**: pass ``bc="periodic"``; ``lbc``/``rbc`` are ignored.

Typical usage::

    from chebfunjax.chebfun1d.pde_solve import pdeSolve
    from chebfunjax.chebfun1d.chebfun import chebfun
    import jax.numpy as jnp
    import numpy as np

    # Heat equation: u_t = u_xx,  u(±1) = 0,  u(x,0) = sin(pi*x)
    u0 = chebfun(lambda x: jnp.sin(jnp.pi * x))
    t_out = np.linspace(0.0, 0.5, 11)
    UU = pdeSolve(lambda t, x, u: u.diff(2), t_out, u0, lbc=0.0, rbc=0.0)

Translated from MATLAB Chebfun ``@chebfun/pdeSolve.m`` (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

Provenance
----------
MATLAB source : @chebfun/pdeSolve.m (1047 lines), @chebfun/pde15s.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

__all__ = ["pdeSolve"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def pdeSolve(
    pdefun: Callable,
    t: "Sequence[float]",
    u0,
    *,
    lbc=None,
    rbc=None,
    bc: str | None = None,
    n: int | None = None,
    n_default: int = 64,
    method: str = "BDF",
    rtol: float = 1e-6,
    atol: float = 1e-8,
):
    """Solve a 1-D PDE using the method of lines with Chebyshev collocation.

    Discretises the spatial part of the PDE on a Chebyshev collocation grid
    of ``n`` points and integrates the resulting large ODE system forward in
    time using ``scipy.integrate.solve_ivp``.  The spatial operator is
    evaluated by applying the user-supplied ``pdefun`` to a
    :class:`~chebfunjax.chebfun1d.Chebfun` constructed from the current
    state vector at each time-step.

    Parameters
    ----------
    pdefun : callable
        Spatial differential operator.  Accepted signatures:

        - ``pdefun(t, x, u)`` — time, space dummy, Chebfun.
        - ``pdefun(t, u)``    — time, Chebfun (``x`` is omitted).

        The function must return a Chebfun (or scalar) representing ``du/dt``.
    t : sequence of float
        Output times.  ``t[0]`` is treated as the initial time; the solver
        integrates from ``t[0]`` to ``t[-1]`` and returns solutions at every
        element of ``t``.
    u0 : Chebfun or callable
        Initial condition.  Either a :class:`~chebfunjax.chebfun1d.Chebfun`
        (whose domain determines the spatial domain) or a callable
        ``u0(x)`` on ``[-1, 1]``.
    lbc : scalar, dict, or None
        Left boundary condition.

        - **Dirichlet**: pass a scalar, e.g. ``lbc=0.0`` imposes ``u(a) = 0``.
        - **Neumann**: pass a dict ``{"neumann": value}``, e.g.
          ``lbc={"neumann": 0.0}`` imposes ``u'(a) = 0``.
        - ``None``: no condition imposed at the left endpoint (periodic or
          pure-Neumann problems must set ``bc="periodic"`` separately).

    rbc : scalar, dict, or None
        Right boundary condition (same format as ``lbc``).
    bc : str or None
        Special BC type.  Currently only ``"periodic"`` is supported.  When
        set, ``lbc`` and ``rbc`` are ignored and periodic boundary conditions
        are enforced by wrapping the ghost point.
    n : int or None
        Fixed collocation grid size.  Default: inferred from ``u0.n`` if
        ``u0`` is a Chebfun, otherwise ``n_default``.
    n_default : int, default 64
        Fallback grid size.
    method : str, default ``"BDF"``
        ODE solver method passed to ``scipy.integrate.solve_ivp``.  ``"BDF"``
        (backward differentiation formulae) mirrors MATLAB's ``ode15s``
        used internally by ``pdeSolve``.  Other choices: ``"Radau"``,
        ``"RK45"``, ``"DOP853"``.
    rtol : float, default 1e-6
        Relative tolerance for the ODE solver.
    atol : float, default 1e-8
        Absolute tolerance for the ODE solver.

    Returns
    -------
    UU : list of Chebfun
        Solutions at each output time in ``t``.  ``UU[k]`` is the Chebfun
        at time ``t[k]``.

    Raises
    ------
    RuntimeError
        If the ODE integrator fails.

    Examples
    --------
    Heat equation: u_t = u_xx with homogeneous Dirichlet BCs.

    >>> import jax.numpy as jnp, numpy as np
    >>> from chebfunjax.chebfun1d.chebfun import chebfun
    >>> from chebfunjax.chebfun1d.pde_solve import pdeSolve
    >>> u0 = chebfun(lambda x: jnp.sin(jnp.pi * x))
    >>> UU = pdeSolve(lambda t, x, u: u.diff(2),
    ...               np.linspace(0, 0.1, 3), u0, lbc=0.0, rbc=0.0)
    >>> len(UU)
    3

    Advection with periodic BCs:

    >>> from chebfunjax.chebfun1d.chebfun import chebfun
    >>> u0p = chebfun(lambda x: jnp.sin(jnp.pi * x))
    >>> UUper = pdeSolve(lambda t, x, u: -u.diff(),
    ...                  np.linspace(0, 0.2, 3), u0p, bc="periodic")
    >>> len(UUper)
    3

    Notes
    -----
    **Boundary condition enforcement**: BCs are applied by the *row
    replacement* strategy (standard in spectral collocation).  After each
    RHS evaluation the DOF(s) corresponding to boundary nodes are replaced by
    the boundary residuals, so the ODE integrator implicitly drives those
    DOFs to satisfy the BC.

    For Dirichlet: row ``i`` of ``dudt`` is replaced by
    ``bc_val - y[i]`` (drives ``y[i] → bc_val``).

    For Neumann: row ``i`` of ``dudt`` is replaced by
    ``bc_val - D[i, :] @ y`` (drives the derivative row to ``bc_val``).

    For periodic BCs: the last grid point is identified with the first;
    the last row is replaced by ``y[0] - y[-1]``.

    Provenance
    ----------
    MATLAB source : @chebfun/pdeSolve.m, @chebfun/pde15s.m
    Chebfun commit: 7574c77

    See Also
    --------
    pde15s : Simpler pde15s wrapper (Radau method).
    """
    import jax.numpy as jnp
    from scipy.integrate import solve_ivp  # type: ignore[import]

    from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
    from chebfunjax.domain import Domain
    from chebfunjax.utils.quadrature import chebpts_ab

    # ----------------------------------------------------------------
    # Parse initial condition and domain
    # ----------------------------------------------------------------
    t_arr = np.asarray(t, dtype=np.float64)

    if isinstance(u0, Chebfun):
        domain = (float(u0.domain.a), float(u0.domain.b))
        if n is None:
            piece_n = u0.funs[0].n if u0.funs else n_default
            n = int(piece_n) if piece_n is not None else n_default
    else:
        domain = (-1.0, 1.0)
        if n is None:
            n = n_default
        u0 = chebfun(u0, domain=domain)

    a, b = domain

    # ----------------------------------------------------------------
    # Build Chebyshev collocation grid and differentiation matrix
    # ----------------------------------------------------------------
    x_ref = np.asarray(chebpts_ab(n, a, b, kind=2))  # physical grid, [a, b]

    # Chebyshev differentiation matrix on reference [-1, 1], then scale
    D1 = _cheb_diff_matrix(n)        # d/dt on [-1, 1]
    scale = 2.0 / (b - a)
    D1_phys = scale * D1             # d/dx on [a, b]

    # ----------------------------------------------------------------
    # Encode initial state
    # ----------------------------------------------------------------
    u0_vals = np.asarray(u0(jnp.array(x_ref)), dtype=np.float64)

    # ----------------------------------------------------------------
    # Determine BC type
    # ----------------------------------------------------------------
    periodic = (bc == "periodic")

    # ----------------------------------------------------------------
    # Build the RHS for the method-of-lines ODE
    # ----------------------------------------------------------------

    def _rhs(t_val: float, y: np.ndarray) -> np.ndarray:
        """Method-of-lines RHS with BC enforcement."""
        # Reconstruct a Chebfun from the current state
        vals = jnp.array(y)
        u_cur = Chebfun.from_values(vals, Domain((a, b)))

        # Evaluate the spatial operator
        try:
            dudt_cheb = pdefun(t_val, None, u_cur)
        except TypeError:
            dudt_cheb = pdefun(t_val, u_cur)

        dudt_vals = np.asarray(dudt_cheb(jnp.array(x_ref)), dtype=np.float64).copy()

        # Enforce boundary conditions
        if periodic:
            # Periodic: identify first and last grid points
            dudt_vals[-1] = y[0] - y[-1]  # drives y[-1] → y[0]
        else:
            # Left BC
            if lbc is not None:
                if isinstance(lbc, dict) and "neumann" in lbc:
                    # Neumann: D1_phys[0, :] @ y == lbc["neumann"]
                    dudt_vals[0] = float(lbc["neumann"]) - float(D1_phys[0, :] @ y)
                else:
                    dudt_vals[0] = float(lbc) - float(y[0])
            # Right BC
            if rbc is not None:
                if isinstance(rbc, dict) and "neumann" in rbc:
                    dudt_vals[-1] = float(rbc["neumann"]) - float(D1_phys[-1, :] @ y)
                else:
                    dudt_vals[-1] = float(rbc) - float(y[-1])

        return dudt_vals

    # ----------------------------------------------------------------
    # Integrate
    # ----------------------------------------------------------------
    t_span_ivp = (float(t_arr[0]), float(t_arr[-1]))

    sol = solve_ivp(
        _rhs,
        t_span_ivp,
        u0_vals,
        method=method,
        t_eval=t_arr,
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(
            f"pdeSolve: ODE solver ({method}) failed: {sol.message}"
        )

    # ----------------------------------------------------------------
    # Reconstruct Chebfun at each output time
    # ----------------------------------------------------------------
    UU = []
    for k in range(sol.y.shape[1]):
        y_k = jnp.array(sol.y[:, k])
        UU.append(Chebfun.from_values(y_k, Domain((a, b))))

    return UU


# ---------------------------------------------------------------------------
# Private helper: Chebyshev differentiation matrix
# ---------------------------------------------------------------------------


def _cheb_diff_matrix(n: int) -> np.ndarray:
    """Return the n×n Chebyshev differentiation matrix on [-1, 1].

    Uses the Chebyshev-2 nodes ``cos(pi*k/(n-1))`` (k = 0, ..., n-1)
    which are in *descending* order (from +1 to -1).

    The matrix D satisfies  D @ f_vals ≈ f'_vals  where ``f_vals[k]``
    is the function value at ``x_k = cos(pi*k/(n-1))``.

    Parameters
    ----------
    n : int
        Grid size (number of nodes).

    Returns
    -------
    D : ndarray, shape (n, n)

    References
    ----------
    Trefethen, "Spectral Methods in MATLAB" (2000), Program 6.
    """
    if n == 0:
        return np.zeros((0, 0))
    if n == 1:
        return np.zeros((1, 1))

    N = n - 1  # polynomial degree
    # Nodes in descending order: x_k = cos(pi*k/N), k=0,...,N
    k = np.arange(N + 1)
    x = np.cos(np.pi * k / N)

    c = np.ones(N + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c *= (-1.0) ** k

    X = np.tile(x, (N + 1, 1))
    dX = X - X.T  # dX[i,j] = x_i - x_j

    # Off-diagonal entries
    D = (c[:, None] / c[None, :]) / np.where(np.abs(dX) < 1e-14, 1.0, dX)
    np.fill_diagonal(D, 0.0)
    np.fill_diagonal(D, -D.sum(axis=1))

    return D
