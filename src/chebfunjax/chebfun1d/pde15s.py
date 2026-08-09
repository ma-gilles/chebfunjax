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

import inspect
from typing import Callable, Sequence

import numpy as np

__all__ = ["pde15s"]


# ---------------------------------------------------------------------------
# Call-signature (arity) dispatch
# ---------------------------------------------------------------------------
#
# MATLAB ``pdeSolve`` normalises both the PDE operator and the boundary-
# condition handles through ``parseFun`` so that, whatever the user wrote,
# they are always invoked as ``op(t, x, u)``.  We mirror that flexibility for
# the Python API: the operator and each BC callable may be written with any of
# ::
#
#     f(u)            # value only          (e.g. ``lambda u: u.diff() + u``)
#     f(t, u)         # time + value        (e.g. ``lambda t, u: u.diff() - g(t)``)
#     f(t, x, u)      # time + space + value (full MATLAB form)
#
# ``_positional_arity`` counts the plain positional parameters so the caller
# can pick the right invocation; genuinely variadic callables fall back to a
# try-chain in :func:`_call_flexible`.


def _positional_arity(fn: Callable) -> int | None:
    """Return the number of positional parameters of *fn* (``None`` if unknown).

    Only ``POSITIONAL_ONLY`` / ``POSITIONAL_OR_KEYWORD`` parameters without a
    ``*args`` catch-all are counted; when the signature cannot be introspected
    (builtins, ``functools.partial`` in some cases, ``*args`` callables) the
    function returns ``None`` and the caller uses a try-chain instead.
    """
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return None
    n = 0
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            return None
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            n += 1
    return n


def _call_flexible(fn: Callable, t, x, u):
    """Invoke *fn* as ``fn(u)``, ``fn(t, u)`` or ``fn(t, x, u)`` as appropriate.

    Dispatch is by positional arity when introspectable; otherwise the three
    forms are attempted in decreasing-arity order and the first that does not
    raise :class:`TypeError` on *argument count* is used.  A :class:`TypeError`
    raised from *inside* ``fn`` (once the argument count matched) propagates
    unchanged.
    """
    arity = _positional_arity(fn)
    if arity == 1:
        return fn(u)
    if arity == 2:
        return fn(t, u)
    if arity is not None:
        # arity 3 (or the degenerate 0/>3): use the full MATLAB form.
        return fn(t, x, u)
    # Unknown arity: try widest-first, skipping argument-count mismatches.
    for args in ((t, x, u), (t, u), (u,)):
        try:
            return fn(*args)
        except TypeError as exc:  # pragma: no cover - defensive
            if "argument" in str(exc):
                continue
            raise
    raise TypeError("pde15s: could not dispatch callable with any known arity.")


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
    lbc : scalar, callable, list, or None
        Left boundary condition(s).  ``None`` means periodic or no condition.
        A scalar imposes Dirichlet ``u(a) = lbc``.  A one-argument callable
        ``lbc(u)`` imposes the homogeneous constraint ``(lbc(u))(a) = 0`` --
        e.g. ``lbc=lambda u: u.diff()`` imposes Neumann ``u'(a) = 0`` and
        ``lbc=lambda u: u - g`` imposes Dirichlet ``u(a) = g``.
    rbc : scalar, callable, list, or None
        Right boundary condition(s); same forms as *lbc* (imposed at ``x=b``).
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
    from chebfunjax.utils.diffmat import diffmat
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

    # Chebyshev collocation grid (2nd kind, ordered a -> b)
    x_ref = np.array(chebpts_ab(n, a, b, kind=2))

    # ----------------------------------------------------------------
    # Encode initial state as a vector
    # ----------------------------------------------------------------
    u0_vals = np.array(u0(jnp.array(x_ref)), dtype=np.float64)

    # ----------------------------------------------------------------
    # Pure-numpy collocation differentiation matrices.
    #
    # The spatial operator is applied through a lightweight proxy
    # (:class:`_MOLProxy`) that represents ``u`` by its nodal values and
    # implements ``u.diff(k)`` as a matrix-vector product with the
    # precomputed Chebyshev-collocation differentiation matrix ``D_k``.  This
    # is *identical* to reconstructing a Chebfun and taking its spectral
    # derivative at the same nodes, but avoids the JAX round-trip on every
    # right-hand-side evaluation (the stiff integrator calls it thousands of
    # times), giving a ~1000x speed-up.
    # ----------------------------------------------------------------
    _dm_cache: dict[int, np.ndarray] = {}

    def _diffmat_k(k: int) -> np.ndarray:
        D = _dm_cache.get(k)
        if D is None:
            D = np.array(diffmat(n, k, domain=(a, b)), dtype=np.float64)
            _dm_cache[k] = D
        return D

    # A proxy representing the spatial coordinate ``x`` on the grid, so that
    # operators / BCs written in the full ``f(t, x, u)`` form can reference it
    # (``x`` is a nodal-value function like any other in the MOL picture).
    x_proxy = _MOLProxy(x_ref, _diffmat_k)

    def _apply_op(t_val: float, y: np.ndarray) -> np.ndarray:
        """Evaluate the spatial operator (no BCs) at state ``y``, time ``t_val``."""
        proxy = _MOLProxy(np.asarray(y, dtype=np.float64), _diffmat_k)
        dudt = _call_flexible(pdefun, t_val, x_proxy, proxy)
        if isinstance(dudt, _MOLProxy):
            return dudt.vals
        return np.broadcast_to(np.asarray(dudt, dtype=np.float64), y.shape).copy()

    # ----------------------------------------------------------------
    # Boundary-condition residuals (driven to zero by row replacement).
    #
    # For a scalar spec ``c`` the condition is Dirichlet ``u(edge) = c``.  For
    # a one-argument callable ``spec(u)`` the condition is ``(spec(u))(edge)=0``
    # (e.g. ``lambda u: u.diff()`` -> Neumann ``u'(edge)=0``).  The MOL row for
    # the boundary node is replaced by ``-residual`` so the stiff integrator
    # enforces the constraint implicitly.
    # ----------------------------------------------------------------
    def _bc_residual(spec, node: int, y: np.ndarray, t_val: float = 0.0) -> float:
        """Residual of one boundary condition at ``node`` (driven to zero).

        Accepts Dirichlet scalars, ``{char, val}`` two-element specs, and the
        general callable residual form.  Callables may be written as ``f(u)``,
        ``f(t, u)`` or ``f(t, x, u)`` (mirroring MATLAB ``pdeSolve``'s
        ``parseFun`` normalisation); the returned function is sampled on the
        grid and its value at the boundary ``node`` is the residual.  This
        covers Dirichlet (``lambda u: u - g``), Neumann (``lambda u: u.diff()``)
        and Robin / flux forms (``lambda u: a*u + b*u.diff() - c``), including
        time-dependent right-hand sides via the ``t`` argument.
        """
        if np.isscalar(spec) or isinstance(spec, (int, float)):
            return float(y[node]) - float(spec)
        if isinstance(spec, (list, tuple)):
            return float(y[node]) - float(spec[0])
        if callable(spec):
            proxy = _MOLProxy(np.asarray(y, dtype=np.float64), _diffmat_k)
            r = _call_flexible(spec, float(t_val), x_proxy, proxy)
            rv = r.vals if isinstance(r, _MOLProxy) else np.asarray(r, dtype=np.float64)
            return float(np.broadcast_to(rv, y.shape)[node])
        raise TypeError(f"pde15s: unrecognised boundary condition {spec!r}.")

    bc_specs = []  # (node_index, spec)
    if lbc is not None:
        bc_specs.append((0, lbc))
    if rbc is not None:
        bc_specs.append((n - 1, rbc))

    def _rhs(t_val: float, y: np.ndarray) -> np.ndarray:
        """RHS of the method-of-lines ODE system (with BC row replacement)."""
        dudt = _apply_op(t_val, y).astype(np.float64, copy=True)
        for node, spec in bc_specs:
            dudt[node] = -_bc_residual(spec, node, y, t_val)
        return dudt

    # ----------------------------------------------------------------
    # Linear, autonomous operator: probe out the spatial operator matrix.
    #
    # All method-of-lines RHS terms in the supported PDEs are linear; probing
    # with the cardinal basis recovers the spatial operator matrix ``L`` and
    # constant term ``c`` exactly (``rhs(y) = L @ y + c``).
    # ----------------------------------------------------------------
    lin = _linearize_operator(_apply_op, n, float(t_arr[0]), float(t_arr[-1]))

    t_span_ivp = (float(t_arr[0]), float(t_arr[-1]))

    bcs_lin = (_linearize_bcs(_bc_residual, bc_specs, n,
                              float(t_arr[0]), float(t_arr[-1]))
               if bc_specs else None)

    if lin is not None and bcs_lin is not None:
        # ------------------------------------------------------------
        # Boundary conditions imposed as algebraic constraints (DAE),
        # matching MATLAB pde15s (which uses ode15s with a singular mass
        # matrix).  The boundary node values are *slaved* to the interior
        # via the linear constraints ``B u = g`` and the interior dynamics
        # are integrated on the reduced (stable) system.  This is essential:
        # imposing the same BCs by row-replacement into the ODE introduces
        # spurious O(1) unstable eigenvalues for advection-dominated PDEs.
        # ------------------------------------------------------------
        L, c = lin
        B, g_func, _bc_time_dep = bcs_lin
        idx_b = [node for node, _ in bc_specs]
        idx_i = [i for i in range(n) if i not in idx_b]
        Bb = B[:, idx_b]
        Bi = B[:, idx_i]
        # u_b = Minv @ u_i + b_off(t)  (slaved boundary values); b_off carries a
        # possibly time-dependent prescribed right-hand side.
        Minv = np.linalg.solve(Bb, -Bi)
        Bb_lu = np.linalg.solve(Bb, np.eye(len(idx_b)))  # Bb^{-1}

        def _b_off(t_val):
            return Bb_lu @ g_func(t_val)

        Li = L[np.ix_(idx_i, idx_i)]
        Lib = L[np.ix_(idx_i, idx_b)]
        A_red = Li + Lib @ Minv
        ci = c[idx_i]

        def _rhs_red(t_val, yi):
            return A_red @ yi + Lib @ _b_off(t_val) + ci

        def _jac_red(_t, _yi):
            return A_red

        yi0 = u0_vals[np.asarray(idx_i)]
        sol = solve_ivp(
            _rhs_red, t_span_ivp, yi0, method=method, t_eval=t_arr,
            rtol=rtol, atol=atol, jac=_jac_red, dense_output=False,
        )

        # Reconstruct the full nodal vectors (interior + slaved boundary),
        # evaluating the prescribed boundary offset at each output time.
        idx_i_arr = np.asarray(idx_i)
        idx_b_arr = np.asarray(idx_b)
        full = np.empty((n, sol.y.shape[1]), dtype=np.float64)
        full[idx_i_arr, :] = sol.y
        b_off_cols = np.column_stack([_b_off(tk) for tk in sol.t])
        full[idx_b_arr, :] = Minv @ sol.y + b_off_cols
    elif lin is not None and not bc_specs:
        # Linear operator with no boundary conditions -> integrate the full
        # system with the analytic Jacobian (fastest path).
        L, c = lin

        def _rhs_full(_t, y):
            return L @ y + c

        def _jac_full(_t, _y):
            return L

        sol = solve_ivp(
            _rhs_full, t_span_ivp, u0_vals, method=method, t_eval=t_arr,
            rtol=rtol, atol=atol, jac=_jac_full, dense_output=False,
        )
        full = np.asarray(sol.y, dtype=np.float64)
    else:
        # Nonlinear operator, or a linear operator with a boundary condition
        # that is genuinely nonlinear / time-dependent (so it cannot be slaved
        # algebraically): enforce the BCs by row replacement and let the solver
        # use its own numerical Jacobian.
        sol = solve_ivp(
            _rhs, t_span_ivp, u0_vals, method=method, t_eval=t_arr,
            rtol=rtol, atol=atol, dense_output=False,
        )
        full = np.asarray(sol.y, dtype=np.float64)

    # A failed integration (stiff front, blow-up, tolerance failure)
    # returns fewer columns than requested output times.  Silently
    # returning the partial list made the CompactingColloids AJR
    # equation look "solved" after 2/101 steps -- surface it loudly.
    if full.shape[1] < len(t_arr):
        import warnings as _warnings

        reached = float(sol.t[-1]) if sol.t.size else float(t_arr[0])
        _warnings.warn(
            "pde15s: time integration stopped early at t="
            f"{reached:.6g} of {float(t_arr[-1]):.6g} "
            f"({full.shape[1]}/{len(t_arr)} output times; solver "
            f"message: {getattr(sol, 'message', 'unknown')}). "
            "Returning the completed steps only.  Consider a larger "
            "n, looser rtol/atol, method='BDF', or a conservative "
            "reformulation for stiff fronts.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ----------------------------------------------------------------
    # Reconstruct Chebfun at each output time
    # ----------------------------------------------------------------
    from chebfunjax.domain import Domain

    UU = []
    for k in range(full.shape[1]):
        y_k = jnp.array(full[:, k])
        UU.append(Chebfun.from_values(y_k, Domain((a, b))))

    return UU


# ---------------------------------------------------------------------------
# Method-of-lines helpers
# ---------------------------------------------------------------------------


class _MOLProxy:
    """Nodal-value proxy for the method-of-lines spatial operator.

    Represents a 1-D function by its values at the ``n`` Chebyshev-2
    collocation nodes.  ``diff(k)`` applies the precomputed ``k``-th
    Chebyshev-collocation differentiation matrix (a pure-numpy mat-vec, hence
    JIT-free and fast), and arithmetic operators act elementwise on the nodal
    values.  Passing this in place of a :class:`~chebfunjax.chebfun1d.Chebfun`
    lets the user's ``pdefun`` be evaluated without any Chebfun round-trip.

    Provenance
    ----------
    MATLAB source : @chebfun/pde15s.m (pdeSolve collocation differentiation)
    Chebfun commit: 7574c77
    """

    __slots__ = ("vals", "_diffmat_k")

    def __init__(self, vals: np.ndarray, diffmat_k) -> None:
        self.vals = np.asarray(vals, dtype=np.float64)
        self._diffmat_k = diffmat_k

    def diff(self, k: int = 1) -> "_MOLProxy":
        if k == 0:
            return self
        return _MOLProxy(self._diffmat_k(k) @ self.vals, self._diffmat_k)

    def _wrap(self, vals) -> "_MOLProxy":
        return _MOLProxy(vals, self._diffmat_k)

    def __add__(self, other):
        o = other.vals if isinstance(other, _MOLProxy) else other
        return self._wrap(self.vals + o)

    __radd__ = __add__

    def __sub__(self, other):
        o = other.vals if isinstance(other, _MOLProxy) else other
        return self._wrap(self.vals - o)

    def __rsub__(self, other):
        o = other.vals if isinstance(other, _MOLProxy) else other
        return self._wrap(o - self.vals)

    def __mul__(self, other):
        o = other.vals if isinstance(other, _MOLProxy) else other
        return self._wrap(self.vals * o)

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = other.vals if isinstance(other, _MOLProxy) else other
        return self._wrap(self.vals / o)

    def __rtruediv__(self, other):
        o = other.vals if isinstance(other, _MOLProxy) else other
        return self._wrap(o / self.vals)

    def __neg__(self):
        return self._wrap(-self.vals)

    def __pow__(self, p):
        return self._wrap(self.vals ** p)


def _linearize_operator(apply_op, n, t0, t1):
    """Recover ``(L, c)`` with ``rhs(y) = L @ y + c`` if the operator is linear.

    Probes the spatial operator with the cardinal basis, then verifies the
    reconstruction against a random state and at both endpoint times.  Returns
    ``None`` for genuinely nonlinear or explicitly time-dependent operators.

    Provenance
    ----------
    MATLAB source : @chebfun/pde15s.m (linear operator detection)
    Chebfun commit: 7574c77
    """
    eye = np.eye(n, dtype=np.float64)
    zero = np.zeros(n, dtype=np.float64)

    c = np.asarray(apply_op(t0, zero), dtype=np.float64).copy()
    L = np.empty((n, n), dtype=np.float64)
    for j in range(n):
        L[:, j] = apply_op(t0, eye[:, j]) - c

    rng = np.random.default_rng(0)
    yr = rng.standard_normal(n)
    pred = L @ yr + c
    ref0 = np.asarray(apply_op(t0, yr), dtype=np.float64)
    ref1 = np.asarray(apply_op(t1, yr), dtype=np.float64)
    scale = max(float(np.max(np.abs(ref0))), 1.0)
    if (np.max(np.abs(ref0 - pred)) > 1e-9 * scale
            or np.max(np.abs(ref1 - ref0)) > 1e-9 * scale):
        return None
    return L, c


def _linearize_bcs(bc_residual, bc_specs, n, t0, t1):
    """Recover the linear boundary constraints ``B @ u = g(t)`` from residuals.

    Each residual ``r_k(t, y) = B[k] @ y - g_k(t)`` must be *affine in* ``y``
    with a *time-invariant slope* ``B`` (Dirichlet, Neumann, or a Robin/flux
    callable such as ``lambda u: a*u + b*u.diff() - c``, optionally with a
    time-dependent right-hand side ``lambda t, u: u - g(t)``).  Probing with the
    cardinal basis at ``t0`` recovers ``B``; the affine reconstruction and the
    time-invariance of ``B`` are verified against a random state at both
    endpoint times.  The constant offset may itself depend on ``t`` (a
    prescribed time-varying boundary value), so the returned ``g`` is a
    *callable* ``g(t)`` giving the length-``nbc`` right-hand side.

    Returns ``None`` when a boundary condition is genuinely nonlinear in ``y``
    (or its slope is time-dependent), so the caller falls back to the
    row-replacement path (which handles those, if only approximately for
    prescribed time-dependent values).

    Provenance
    ----------
    MATLAB source : @chebfun/pde15s.m (boundary condition assembly)
    Chebfun commit: 7574c77
    """
    eye = np.eye(n, dtype=np.float64)
    zero = np.zeros(n, dtype=np.float64)
    nbc = len(bc_specs)
    B = np.empty((nbc, n), dtype=np.float64)
    rng = np.random.default_rng(1)
    yr = rng.standard_normal(n)
    time_dependent = False
    for k, (node, spec) in enumerate(bc_specs):
        r0 = bc_residual(spec, node, zero, t0)
        for j in range(n):
            B[k, j] = bc_residual(spec, node, eye[:, j], t0) - r0
        # Slope at the far endpoint time (must match B for a valid slaving).
        r0_t1 = bc_residual(spec, node, zero, t1)
        if abs(r0_t1 - r0) > 1e-12 * max(abs(r0), 1.0):
            time_dependent = True
        # Verify affinity (linear in y) and time-invariance of the slope B[k].
        pred = float(B[k] @ yr + r0)
        actual0 = float(bc_residual(spec, node, yr, t0))
        actual1_slope = float(bc_residual(spec, node, yr, t1)) - (r0_t1 - r0)
        scale = max(abs(actual0), 1.0)
        if (abs(actual0 - pred) > 1e-9 * scale
                or abs(actual1_slope - actual0) > 1e-9 * scale):
            # Nonlinear in y, or the slope B varies with t: not slave-able.
            return None

    def g(t_val: float) -> np.ndarray:
        return np.array(
            [-bc_residual(spec, node, zero, float(t_val))
             for (node, spec) in bc_specs],
            dtype=np.float64,
        )

    return B, g, time_dependent
