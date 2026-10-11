"""Polynomial pde15s with JAX NDF and native spatial adaptation by default.

Source: Chebfun 7574c77 pdeSolve and MATLAB R2025b ode15s/daeic12.
The NDF path supports a single real polynomial piece and constant full mass.
Two real components support one handle residual per component on each boundary.
Larger systems, Fourier PDE evolution, complex, sparse, and variable-mass NDF remain unsupported.
Smooth pieces are merged; trigonometric initial data uses polynomial collocation.

Provenance
----------
Chebfun @chebfun/pde15s.m and pdeSolve.m; MATLAB R2025b ode15s.m.
Chebfun native commit: 7574c77680d7e82b79626300bf255498271a72df.
Native ODE runtime/source release: MATLAB R2025b.
"""

from ._pde_ndf.arity import _call_flexible, _positional_arity
from ._pde_ndf.pde_driver import solve

__all__ = ["pde15s", "_call_flexible", "_positional_arity"]


def pde15s(
    pdefun, t, u0, *, lbc=None, rbc=None, n=None, n_default=64, method="NDF", rtol=1e-6, atol=1e-6
):
    """Return scalar or two-component array-valued Chebfuns at the supplied increasing output times.

    The default uses native spatial adaptation; n selects native fixed-N
    coefficient prolongation and outputs. Omitting both boundary conditions
    preserves the Python unconstrained nodal evolution extension, not periodic
    Fourier semantics. Two supplied times remain two outputs in this API.

    method='NDF' uses JAX only and native default tolerances (1e-6 for both).
    Explicit historical methods retain the existing SciPy compatibility path;
    those paths remain outstanding non-JAX work. Two-component callbacks accept
    (u,v) or (t,x,u,v) and return two expressions; both boundary handles
    must return two residuals. Larger systems, Fourier PDE evolution, complex,
    sparse and variable-mass equations remain unsupported by the NDF path.

    Provenance
    ----------
    MATLAB @chebfun/pde15s.m and pdeSolve.m; R2025b ode15s.m.
    Chebfun commit: 7574c77
    """
    if method != "NDF":
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
        pdefun, t, u0, lbc=lbc, rbc=rbc, rtol=rtol, atol=atol, fixed_n=n, record_trace=False
    )[0]
