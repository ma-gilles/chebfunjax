"""chebfunjax — Chebfun in Python, powered by JAX."""

__version__ = "0.1.0"

# Enable float64 by default — spectral methods need double precision
import jax

jax.config.update("jax_enable_x64", True)

# Public API
from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun  # noqa: E402
from chebfunjax.chebfun2d.chebfun2 import Chebfun2, chebfun2  # noqa: E402

# Plotting — importable as cj.plot(f), cj.surf(g), etc.
from chebfunjax.plotting import (  # noqa: E402
    plot,
    plotcoeffs,
    contour,
    surf,
    phaseplot,
    plot_disk,
    plot_sphere,
    plot_slices,
)

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
    # Plotting
    "plot",
    "plotcoeffs",
    "contour",
    "surf",
    "phaseplot",
    "plot_disk",
    "plot_sphere",
    "plot_slices",
]
