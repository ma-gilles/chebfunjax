"""MATLAB string-expression constructor support.

Translates MATLAB anonymous-expression strings like
``'cos(x) + sin(x.*y)'`` into vectorized JAX callables so the string
constructor syntaxes ``chebfun2('...')`` / ``chebfun3('...')`` work.

Provenance
----------
MATLAB source : chebfun2/chebfun3 constructor string parsing
    (@chebfun2/chebfun2.m 'str2op')
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _source_log(x):
    """MATLAB numeric log, including eager negative-real continuation.

    Eager values preserve MATLAB's data-dependent real/complex dtype.
    Real traced values retain the fixed-real JAX contract: negative traced
    values produce NaN; pass complex inputs for traced complex continuation.
    This avoids silently changing positive-domain traced output storage.

    Provenance
    ----------
    MATLAB source: numeric log used by @chebfun/chebfun.m string operators.
    Chebfun commit: 7574c77.
    Eager dtype dispatch follows utils/_ratbary.py's source log fallback.
    """
    x = jnp.asarray(x)
    if not isinstance(x, jax.core.Tracer) and not jnp.iscomplexobj(x):
        if bool(jax.device_get(jnp.any(x < 0))):
            x = x.astype(jnp.result_type(x, 1j))
    return jnp.log(x)


_FUNS = {name: getattr(jnp, jname) for name, jname in [
    ("sin", "sin"), ("cos", "cos"), ("tan", "tan"), ("exp", "exp"),
    ("log", "log"), ("sqrt", "sqrt"), ("abs", "abs"),
    ("sinh", "sinh"), ("cosh", "cosh"), ("tanh", "tanh"),
    ("asin", "arcsin"), ("acos", "arccos"), ("atan", "arctan"),
]}
_FUNS["log"] = _source_log
_FUNS["pi"] = jnp.pi
# MATLAB numeric heaviside has half value at zero; source @chebfun/
# heaviside.m likewise uses 0.5*(sign(f)+1). Register the name so
# _string_op does not mistake it for the expression variable.
_FUNS["heaviside"] = lambda x: jnp.heaviside(x, 0.5)


def matlab_expression(expr: str, var_names: tuple[str, ...]):
    """Compile a MATLAB expression string into a vectorized callable
    of ``var_names`` (elementwise operators translated)."""
    s = expr.replace(".*", "*").replace("./", "/")
    s = s.replace(".^", "**").replace("^", "**")
    ns = {"__builtins__": {}}
    ns.update(_FUNS)
    try:
        fn = eval(  # noqa: S307 -- restricted namespace, math only
            f"lambda {', '.join(var_names)}: {s}", ns)
    except SyntaxError as exc:
        raise TypeError(
            f"Cannot construct a chebfun from the string {expr!r}: not a "
            "valid MATLAB expression.") from exc
    return fn
