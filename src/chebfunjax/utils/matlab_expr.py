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

import ast

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


def _eager_real_storage(value):
    """Normalize exact-real eager multiplication/division output storage.

    Provenance
    ----------
    MATLAB R2025b numeric primitive capture pnx2: complex-zero input storage
    remains complex, while multiplication/division by real ones is real.
    Chebfun commit: 7574c77 (string-constructor caller).
    Traced outputs keep their static JAX dtype; no imaginary tolerance used.
    """
    value = jnp.asarray(value)
    if not isinstance(value, jax.core.Tracer) and jnp.iscomplexobj(value):
        if bool(jax.device_get(jnp.all(jnp.imag(value) == 0))):
            return jnp.real(value)
    return value


def _source_multiply(left, right):
    return _eager_real_storage(left * right)


def _source_divide(left, right):
    return _eager_real_storage(left / right)


class _NumericArithmetic(ast.NodeTransformer):
    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        name = {ast.Mult: "_source_multiply", ast.Div: "_source_divide"}.get(type(node.op))
        if name is None:
            return node
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=name, ctx=ast.Load()), args=[node.left, node.right], keywords=[]
            ),
            node,
        )


_FUNS = {
    name: getattr(jnp, jname)
    for name, jname in [
        ("sin", "sin"),
        ("cos", "cos"),
        ("tan", "tan"),
        ("exp", "exp"),
        ("log", "log"),
        ("sqrt", "sqrt"),
        ("abs", "abs"),
        ("sinh", "sinh"),
        ("cosh", "cosh"),
        ("tanh", "tanh"),
        ("asin", "arcsin"),
        ("acos", "arccos"),
        ("atan", "arctan"),
    ]
}
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
    ns.update(_source_multiply=_source_multiply, _source_divide=_source_divide)
    try:
        tree = ast.parse(f"lambda {', '.join(var_names)}: {s}", mode="eval")
        tree = ast.fix_missing_locations(_NumericArithmetic().visit(tree))
        fn = eval(compile(tree, "<MATLAB expression>", "eval"), ns)  # noqa: S307
    except SyntaxError as exc:
        raise TypeError(
            f"Cannot construct a chebfun from the string {expr!r}: not a valid MATLAB expression."
        ) from exc
    return fn
