"""Existing Python pde15s callable-arity compatibility helpers.

Provenance
----------
Python callable-arity compatibility adaptation of Chebfun pdeSolve.m parseFun.
The preexisting Python helper bodies are retained.
Chebfun native commit: 7574c77680d7e82b79626300bf255498271a72df.
Native ODE runtime/source release: MATLAB R2025b.
"""

import inspect
from typing import Callable


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
