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

import jax.numpy as jnp

_FUNS = {name: getattr(jnp, jname) for name, jname in [
    ("sin", "sin"), ("cos", "cos"), ("tan", "tan"), ("exp", "exp"),
    ("log", "log"), ("sqrt", "sqrt"), ("abs", "abs"),
    ("sinh", "sinh"), ("cosh", "cosh"), ("tanh", "tanh"),
    ("asin", "arcsin"), ("acos", "arccos"), ("atan", "arctan"),
]}
_FUNS["pi"] = jnp.pi


def matlab_expression(expr: str, var_names: tuple[str, ...]):
    """Compile a MATLAB expression string into a vectorized callable
    of ``var_names`` (elementwise operators translated)."""
    s = expr.replace(".*", "*").replace("./", "/")
    s = s.replace(".^", "**").replace("^", "**")
    ns = {"__builtins__": {}}
    ns.update(_FUNS)
    fn = eval(  # noqa: S307 -- restricted namespace, math only
        f"lambda {', '.join(var_names)}: {s}", ns)
    return fn
