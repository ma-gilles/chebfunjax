"""chebfunjax — Chebfun in Python, powered by JAX."""

__version__ = "0.1.0"

# Enable float64 by default — spectral methods need double precision
import jax

jax.config.update("jax_enable_x64", True)

# Public API
from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun  # noqa: E402
from chebfunjax.chebfun2d.chebfun2 import Chebfun2, chebfun2  # noqa: E402

# Random functions
from chebfunjax.utils.random import randnfun  # noqa: E402

# Plotting — importable as cj.plot(f), cj.surf(g), etc.
from chebfunjax.plotting import (  # noqa: E402
    plot_1d,
    plot_dispatch,
    plotcoeffs,
    contour,
    surf,
    phaseplot,
    plot_disk,
    plot_sphere,
    plot_slices,
    quiver_sphere,
    isosurface_ball,
    # New rich plotting API
    waterfall,
    roots_plot,
    spy,
    plotregion,
    arrowplot,
    chebpolyplot,
    # Style
    chebfun_style,
    CHEBFUN_RC,
    PARULA,
)

# The top-level `plot` is the universal dispatcher (like MATLAB's plot(f))
plot = plot_dispatch

# ---------------------------------------------------------------------------
# Top-level special functions: cj.sin(f) is equivalent to f.sin()
# ---------------------------------------------------------------------------

def sin(f: Chebfun) -> Chebfun:
    """Sine of a Chebfun.  Equivalent to ``f.sin()``."""
    return f.sin()


def cos(f: Chebfun) -> Chebfun:
    """Cosine of a Chebfun.  Equivalent to ``f.cos()``."""
    return f.cos()


def exp(f: Chebfun) -> Chebfun:
    """Exponential of a Chebfun.  Equivalent to ``f.exp()``."""
    return f.exp()


def log(f: Chebfun) -> Chebfun:
    """Natural logarithm of a Chebfun.  Equivalent to ``f.log()``."""
    return f.log()


def sqrt(f: Chebfun) -> Chebfun:
    """Square root of a Chebfun.  Equivalent to ``f.sqrt()``."""
    return f.sqrt()


def abs(f: Chebfun) -> Chebfun:  # noqa: A001 (shadows builtin intentionally)
    """Absolute value of a Chebfun.  Equivalent to ``f.abs()``."""
    return f.abs()


def sign(f: Chebfun) -> Chebfun:
    """Sign function of a Chebfun.  Equivalent to ``f.sign()``."""
    return f.sign()


def sinh(f: Chebfun) -> Chebfun:
    """Hyperbolic sine of a Chebfun.  Equivalent to ``f.sinh()``."""
    return f.sinh()


def cosh(f: Chebfun) -> Chebfun:
    """Hyperbolic cosine of a Chebfun.  Equivalent to ``f.cosh()``."""
    return f.cosh()


def tanh(f: Chebfun) -> Chebfun:
    """Hyperbolic tangent of a Chebfun.  Equivalent to ``f.tanh()``."""
    return f.tanh()


def asin(f: Chebfun) -> Chebfun:
    """Inverse sine of a Chebfun.  Equivalent to ``f.asin()``."""
    return f.asin()


def acos(f: Chebfun) -> Chebfun:
    """Inverse cosine of a Chebfun.  Equivalent to ``f.acos()``."""
    return f.acos()


def atan(f: Chebfun) -> Chebfun:
    """Inverse tangent of a Chebfun.  Equivalent to ``f.atan()``."""
    return f.atan()


def atan2(y: Chebfun, x: Chebfun) -> Chebfun:
    """Four-quadrant arctangent.  See :func:`chebfunjax.chebfun1d.chebfun.atan2`."""
    from chebfunjax.chebfun1d.chebfun import atan2 as _atan2
    return _atan2(y, x)


# New functions

def besselh(f: Chebfun, nu: float, k: int = 1, *, scale: int = 0):
    """Hankel function of f.  Returns ``(H_re, H_im)`` pair of Chebfuns.  Equivalent to ``f.besselh(nu, k, scale=scale)``."""
    return f.besselh(nu, k, scale=scale)


def besselk(f: Chebfun, nu: float, *, scale: int = 0) -> Chebfun:
    """Modified Bessel K of f.  Equivalent to ``f.besselk(nu, scale=scale)``."""
    return f.besselk(nu, scale=scale)


def ellipke(f: Chebfun):
    """Complete elliptic integrals K(f), E(f).  Equivalent to ``f.ellipke()``."""
    return f.ellipke()


def dirac(f: Chebfun) -> Chebfun:
    """Dirac delta at roots of f.  Equivalent to ``f.dirac()``."""
    return f.dirac()


def unwrap(f: Chebfun, jump_tol=None) -> Chebfun:
    """Phase-unwrap f.  Equivalent to ``f.unwrap(jump_tol)``."""
    return f.unwrap(jump_tol)


def iszero(f: Chebfun) -> bool:
    """True if f is identically zero.  Equivalent to ``f.iszero()``."""
    return f.iszero()


def innerProduct(f: Chebfun, g: Chebfun):
    """L2 inner product.  Equivalent to ``f.inner(g)``."""
    from chebfunjax.chebfun1d.chebfun import innerProduct as _ip
    return _ip(f, g)


def lagrange(x, domain=None):
    """Lagrange basis polynomials.  See :func:`chebfunjax.chebfun1d.chebfun.lagrange`."""
    from chebfunjax.chebfun1d.chebfun import lagrange as _lag
    return _lag(x, domain)


def subspace(A, B) -> float:
    """Principal angle between quasimatrix subspaces.  See :func:`chebfunjax.chebfun1d.chebfun.subspace`."""
    from chebfunjax.chebfun1d.chebfun import subspace as _sub
    return _sub(A, B)


def quantumstates(V: Chebfun, n: int = 10, h: float = 0.1):
    """Quantum eigenstates.  See :func:`chebfunjax.chebfun1d.chebfun.quantumstates`."""
    from chebfunjax.chebfun1d.chebfun import quantumstates as _qs
    return _qs(V, n, h)


def ode78(odefun, tspan, y0, **kwargs):
    """7(8)-order ODE integrator.  See :func:`chebfunjax.chebfun1d.chebfun.ode78`."""
    from chebfunjax.chebfun1d.chebfun import ode78 as _ode78
    return _ode78(odefun, tspan, y0, **kwargs)


def ode89(odefun, tspan, y0, **kwargs):
    """8(9)-order ODE integrator.  See :func:`chebfunjax.chebfun1d.chebfun.ode89`."""
    from chebfunjax.chebfun1d.chebfun import ode89 as _ode89
    return _ode89(odefun, tspan, y0, **kwargs)


def pdeSolve(pdefun, t, u0, **kwargs):
    """Method-of-lines PDE solver.  See :func:`chebfunjax.chebfun1d.pde_solve.pdeSolve`."""
    from chebfunjax.chebfun1d.pde_solve import pdeSolve as _pdeSolve
    return _pdeSolve(pdefun, t, u0, **kwargs)


__all__ = [
    "Chebfun",
    "chebfun",
    "Chebfun2",
    "chebfun2",
    # Special functions
    "sin",
    "cos",
    "exp",
    "log",
    "sqrt",
    "abs",
    "sign",
    "sinh",
    "cosh",
    "tanh",
    "asin",
    "acos",
    "atan",
    "atan2",
    # Bessel and special functions
    "besselh",
    "besselk",
    "ellipke",
    "dirac",
    "unwrap",
    "iszero",
    "innerProduct",
    # Interpolation / linear algebra
    "lagrange",
    "subspace",
    # Quantum
    "quantumstates",
    # ODE integrators
    "ode78",
    "ode89",
    # PDE solver
    "pdeSolve",
    # Random functions
    "randnfun",
    # Plotting
    "plot",
    "plot_1d",
    "plot_dispatch",
    "plotcoeffs",
    "contour",
    "surf",
    "phaseplot",
    "plot_disk",
    "plot_sphere",
    "plot_slices",
    "quiver_sphere",
    "isosurface_ball",
    "waterfall",
    "roots_plot",
    "spy",
    "plotregion",
    "arrowplot",
    "chebpolyplot",
    "PARULA",
]
