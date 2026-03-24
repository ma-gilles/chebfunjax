# uses-numpy: ODE integration via scipy/numpy (not JIT-safe by design)
"""Method-of-lines PDE solver for 1-D PDEs.

Discretizes in space using a Chebyshev spectral collocation grid and
integrates the resulting ODE system in time with ``scipy.integrate.solve_ivp``
(default method: ``"Radau"`` — a stiff solver, matching MATLAB's ``pde15s``
which uses ``ode15s``).

Typical usage::

    from chebfunjax.chebfun1d.pde15s import pde15s
    import jax.numpy as jnp

    # Heat equation: u_t = u_xx,  u(±1) = 0,  u(x,0) = sin(pi*x)
    u0 = chebfun(lambda x: jnp.sin(jnp.pi * x))
    t_span = (0.0, 0.5)
    t_out  = jnp.linspace(0.0, 0.5, 11)
    UU = pde15s(lambda t, x, u: u.diff(2), t_out, u0, lbc=0.0, rbc=0.0)

Translated from MATLAB Chebfun ``@chebfun/pde15s.m`` (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

Provenance
----------
MATLAB source : @chebfun/pde15s.m, pdeSolve.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

__all__ = ["pde15s"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def pde15s(
    pdefun: Callable,
    t: Sequence[float],
    u0,
    *,
    lbc=None,
    rbc=None,
    n: int | None = None,
    n_default: int = 64,
    method: str = "Radau",
    rtol: float = 1e-6,
    atol: float = 1e-8,
):
    """Solve a PDE using the method of lines with Chebyshev collocation in space.

    Discretises the spatial part of the PDE on a Chebyshev collocation grid
    of ``n`` points and integrates the resulting large ODE system forward in
    time using ``scipy.integrate.solve_ivp``.  The spatial operator is
    evaluated by applying the user-supplied ``pdefun`` to a
    :class:`~chebfunjax.chebfun1d.Chebfun` constructed from the current
    state vector at each time-step.

    Parameters
    ----------
    pdefun : callable
        Spatial differential operator.  Signature ``pdefun(t, x, u)`` where
        *t* is the current time (float), *x* is a dummy variable (not used
        directly — the chebfun ``u`` already knows its domain), and ``u`` is
        a :class:`~chebfunjax.chebfun1d.Chebfun`.  Must return a Chebfun
        representing ``du/dt``.  Alternatively, the two-argument form
        ``pdefun(t, u)`` is also accepted.
    t : sequence of float
        Output times.  ``t[0]`` is treated as the initial time; the solver
        integrates from ``t[0]`` to ``t[-1]`` and returns solutions at every
        element of *t*.
    u0 : Chebfun or callable
        Initial condition.  Either a :class:`~chebfunjax.chebfun1d.Chebfun`
        (whose domain determines the spatial domain) or a callable
        ``u0(x)`` on ``[-1, 1]``.
    lbc : scalar, list, or None
        Left boundary condition(s).  ``None`` means periodic or no condition.
        Scalars impose Dirichlet: ``u(a) = lbc``.
    rbc : scalar, list, or None
        Right boundary condition(s).
    n : int or None
        Fixed collocation grid size.  Default: inferred from ``u0.n`` if
        ``u0`` is a Chebfun, otherwise ``n_default``.
    n_default : int, default 64
        Fallback grid size when ``n`` is ``None`` and cannot be inferred.
    method : str, default ``"Radau"``
        ODE solver method passed to ``scipy.integrate.solve_ivp``.  ``"Radau"``
        (a stiff implicit Runge-Kutta method) mirrors MATLAB ``ode15s``.
        Other valid choices: ``"BDF"``, ``"RK45"``, ``"DOP853"``.
    rtol : float, default 1e-6
        Relative tolerance for the ODE solver.
    atol : float, default 1e-8
        Absolute tolerance for the ODE solver.

    Returns
    -------
    UU : list of Chebfun
        Solutions at each output time in *t*.  ``UU[k]`` is the Chebfun at
        ``t[k]``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> from chebfunjax.chebfun1d.chebfun import chebfun
    >>> from chebfunjax.chebfun1d.pde15s import pde15s
    >>> # Heat equation: u_t = 0.1 * u_xx,  u(±1)=0,  u(x,0)=sin(pi*x)
    >>> u0 = chebfun(lambda x: jnp.sin(jnp.pi * x))
    >>> UU = pde15s(lambda t, x, u: 0.1 * u.diff(2),
    ...             np.linspace(0, 0.1, 3), u0, lbc=0.0, rbc=0.0)
    >>> len(UU)
    3

    Notes
    -----
    Boundary conditions are enforced by *row replacement*: after each
    right-hand-side evaluation the rows of the spatial-derivative matrix
    corresponding to the boundary nodes are replaced by the boundary
    residuals, driving the ODE integrator to enforce them implicitly.
    This is the classical Chebfun / spectral collocation approach.

    For strongly nonlinear PDEs or PDEs with sharp fronts, consider
    using a finer grid (larger ``n``) or a smaller ``atol``/``rtol``.

    Provenance
    ----------
    MATLAB source : @chebfun/pde15s.m, pdeSolve.m
    Chebfun commit: 7574c77

    See Also
    --------
    bvp, ivp
    """
    import jax.numpy as jnp
    from scipy.integrate import solve_ivp

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
            # Use the degree of the first piece
            piece_n = u0.funs[0].n if hasattr(u0.funs[0], "n") else n_default
            n = int(piece_n) if piece_n is not None else n_default
    else:
        domain = (-1.0, 1.0)
        if n is None:
            n = n_default
        u0 = chebfun(u0, domain=domain)

    a, b = domain

    # Chebyshev collocation grid (2nd kind, ordered -1 to 1 on reference)
    x_ref = np.array(chebpts_ab(n, a, b, kind=2))

    # ----------------------------------------------------------------
    # Encode initial state as a vector
    # ----------------------------------------------------------------
    u0_vals = np.array(u0(jnp.array(x_ref)))

    # ----------------------------------------------------------------
    # Build the RHS for the ODE system
    # ----------------------------------------------------------------

    def _rhs(t_val: float, y: np.ndarray) -> np.ndarray:
        """RHS of the method-of-lines ODE system."""
        # Reconstruct a Chebfun from the current state vector
        vals = jnp.array(y)
        u_cur = Chebfun.from_values(vals, Domain((a, b)))

        # Evaluate the spatial operator
        try:
            dudt = pdefun(t_val, None, u_cur)
        except TypeError:
            dudt = pdefun(t_val, u_cur)

        dudt_vals = np.array(dudt(jnp.array(x_ref)))

        # Enforce boundary conditions by row replacement
        if lbc is not None:
            lbc_val = float(lbc) if np.isscalar(lbc) else float(lbc[0])
            dudt_vals[0] = lbc_val - float(y[0])
        if rbc is not None:
            rbc_val = float(rbc) if np.isscalar(rbc) else float(rbc[0])
            dudt_vals[-1] = rbc_val - float(y[-1])

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

    # ----------------------------------------------------------------
    # Reconstruct Chebfun at each output time
    # ----------------------------------------------------------------
    UU = []
    for k in range(sol.y.shape[1]):
        y_k = jnp.array(sol.y[:, k])
        UU.append(Chebfun.from_values(y_k, Domain((a, b))))

    return UU
