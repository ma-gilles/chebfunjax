"""Polynomial pde15s with JAX NDF and native spatial adaptation by default.

Source: Chebfun 7574c77 pdeSolve and MATLAB R2025b ode15s/daeic12.
The NDF path supports a single real polynomial piece and constant full mass.
Real multicomponent systems count the residuals of each supplied boundary handle.
Fourier PDE evolution, complex, sparse, and variable-mass NDF remain unsupported.
Smooth pieces are merged; trigonometric initial data uses polynomial collocation.

Provenance
----------
Chebfun @chebfun/pde15s.m and pdeSolve.m; MATLAB R2025b ode15s.m.
Chebfun native commit: 7574c77680d7e82b79626300bf255498271a72df.
Native ODE runtime/source release: MATLAB R2025b.
"""

import math
from collections.abc import Mapping

import jax.numpy as jnp

from ._pde_ndf.arity import _call_flexible, _positional_arity
from ._pde_ndf.linear_boundary import _boundary_token
from ._pde_ndf.pde_driver import solve
from .pdeset import ODE_FIELDS, PDE_FIELDS, _as_matlab_logical, _empty

__all__ = ["pde15s", "_call_flexible", "_positional_arity"]


def _consumer_options(options):
    """Read supported pdeSolve fields; never silently discard solver options."""
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError("PDE options must be a pdeset mapping")
    supported = {"Eps", "N", "RelTol", "AbsTol", "AdjustBCs", "ODESolver"}
    known = set(ODE_FIELDS + PDE_FIELDS)
    parsed = {}
    for name, value in options.items():
        if _empty(value):
            continue
        if name not in known:
            raise NotImplementedError(f"PDE option {name!r} is not implemented")
        if name in supported:
            parsed[name] = value
        elif name == "Plot" and not (isinstance(value, str) and value.lower() == "on"):
            # pdeSolve uses strcmpi(value, 'on'), not logical conversion.
            continue
        elif name in {"HoldPlot", "YLim", "PlotStyle"}:
            # These fields have no effect when source plotting is disabled.
            continue
        else:
            raise NotImplementedError(f"PDE option {name!r} is not implemented")
    return parsed


def pde15s(
    pdefun,
    t,
    u0,
    bc=None,
    options=None,
    spatial_n=None,
    *,
    lbc=None,
    rbc=None,
    n=None,
    n_default=64,
    method="NDF",
    rtol=None,
    atol=None,
):
    """Return scalar or multicomponent array-valued Chebfuns at the supplied increasing output times.

    The default uses native spatial adaptation; n selects native fixed-N
    coefficient prolongation and outputs. For scalar inputs, omitting both boundary conditions
    preserves the Python unconstrained nodal evolution extension, not periodic
    Fourier semantics. Two supplied times remain two outputs in this API.

    ``options`` accepts a pdeset mapping in the fifth positional slot. Eps
    controls spatial adaptation and supplies both default ODE tolerances;
    RelTol/AbsTol override those defaults. N selects fixed spatial size and
    a sixth positional size overrides options.N. Explicit Python n/rtol/atol
    keywords take precedence. AdjustBCs=False disables initial boundary
    offsets and their warning. As in native pde15s, options.ODESolver is
    overridden by this wrapper. Plot='on' and additional solver controls
    are not yet implemented and raise rather than being silently ignored.

    bc='neumann' or bc='dirichlet' supplies symmetric source linear endpoint
    conditions, one row per component at each end. The same tokens can be
    supplied separately as lbc and rbc; callbacks retain their residual counts.

    method='NDF' uses JAX only and native default tolerances (1e-6 for both).
    Explicit historical methods retain the existing SciPy compatibility path;
    those paths remain outstanding non-JAX work. N-component callbacks accept
    the N dependent components, optionally preceded by one independent input
    or (t,x), and return N
    expressions. With N+1 inputs, the independent argument is space for the
    PDE and time for endpoint boundary handles. Each supplied boundary handle
    returns a tuple or list of residuals; an omitted side adds no residuals.
    Their total count must match the sum of the
    differential orders of the PDE expressions. Fourier PDE evolution,
    complex, sparse and variable-mass equations remain unsupported by the NDF path.

    Provenance
    ----------
    MATLAB @chebfun/pde15s.m and pdeSolve.m; R2025b ode15s.m.
    Chebfun commit: 7574c77
    """
    opt = _consumer_options(options)
    if options is not None and method != "NDF":
        raise NotImplementedError("pdeset consumers require method='NDF'")
    spatial_tol = opt.get("Eps", 1e-6)
    rtol = opt.get("RelTol", spatial_tol) if rtol is None else rtol
    atol = opt.get("AbsTol", spatial_tol) if atol is None else atol
    # ODE options permit a componentwise AbsTol vector. Keep division and
    # threshold construction in JAX even when a caller supplied a Python list.
    rtol = jnp.asarray(rtol)
    atol = jnp.asarray(atol)
    if spatial_n is not None:
        if n is not None:
            raise ValueError("Use a positional spatial size or n, not both")
        n = spatial_n
    if n is None:
        n = opt.get("N")
        if n is not None and math.isnan(float(n)):
            n = None
    adjust_bcs = _as_matlab_logical(opt.get("AdjustBCs", True))
    if bc is not None:
        if method != "NDF":
            raise NotImplementedError("Boundary shorthand is implemented for method='NDF'")
        if lbc is not None or rbc is not None:
            raise ValueError("Use bc or explicit lbc/rbc, not both")
        lbc = rbc = _boundary_token(bc)
    if method != "NDF":
        if any(isinstance(spec, str) for spec in (lbc, rbc)):
            raise NotImplementedError("Endpoint shorthand is implemented for method='NDF'")
        from ._pde_legacy import pde15s as legacy_pde15s

        return legacy_pde15s(
            pdefun,
            t,
            u0,
            lbc=lbc,
            rbc=rbc,
            n=n,
            n_default=n_default,
            method=method,
            rtol=rtol,
            atol=atol,
        )
    if callable(u0) and not hasattr(u0, "funs"):
        from .chebfun import chebfun

        u0 = chebfun(u0)
        if n is None:
            n = n_default
    return solve(
        pdefun, t, u0, lbc=lbc, rbc=rbc, rtol=rtol, atol=atol,
        spatial_tol=spatial_tol, fixed_n=n, adjust_bcs=adjust_bcs, record_trace=False
    )[0]
