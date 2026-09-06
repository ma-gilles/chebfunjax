"""Parsing of MATLAB-style ``spinop2``/``spinop3`` operator handles.

MATLAB users write the linear and nonlinear parts of a 2-D/3-D SPINOP as
anonymous functions, e.g. ``S.lin = @(u,v) [2e-5*lap(u); 1e-5*lap(v)]``
and ``S.nonlin = @(u,v) [F*(1-u)-u.*v.^2; -(F+K)*v+u.*v.^2]``; MATLAB's
``discretize`` turns the linear handle into a diagonal Fourier symbol via
``func2str`` and a trigspec CHEBOP.  chebfunjax accepts the same text: the
linear part must be a linear combination of ``lap``, ``biharm``,
``triharm``, ``quadharm`` and ``quintharm`` of each unknown (the operators
SPINOP2/SPINOP3 support), the nonlinear part any elementwise MATLAB
expression of the unknowns.

Provenance
----------
MATLAB source : @spinoperator/discretize.m (func2str parsing),
    @spinop2/spinop2.m, @spinop3/spinop3.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and The
    Chebfun Developers.
"""

from __future__ import annotations

import re

_OPS = ("lap", "biharm", "triharm", "quadharm", "quintharm")
_OP_INDEX = {op: k for k, op in enumerate(_OPS)}


def _split_handle(text: str):
    """``'@(u,v) [a; b]'`` -> (('u', 'v'), ['a', 'b'])."""
    s = text.strip()
    m = re.match(r"^@\(\s*([^)]*)\)\s*(.*)$", s, flags=re.S)
    if not m:
        raise ValueError(
            f"Expected a MATLAB anonymous function '@(u, ...) ...', got {text!r}.")
    names = tuple(v.strip() for v in m.group(1).split(",") if v.strip())
    body = m.group(2).strip()
    if body.startswith("[") and body.endswith("]"):
        body = body[1:-1]
    parts = _split_top(body, ";")
    return names, [p.strip() for p in parts if p.strip()]


def _split_top(s: str, sep: str):
    """Split on ``sep`` at parenthesis/bracket depth 0."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def _terms(expr: str):
    """Top-level additive terms of ``expr`` with their signs."""
    s = re.sub(r"(\d)[eE]-(\d)", r"\1e_m\2", expr)   # protect exponents
    s = re.sub(r"(\d)[eE]\+(\d)", r"\1e_p\2", s)
    pieces, depth, cur = [], 0, []
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch in "+-" and depth == 0 and cur and "".join(cur).strip():
            pieces.append("".join(cur))
            cur = [ch]
        else:
            cur.append(ch)
    pieces.append("".join(cur))
    return [p.replace("e_m", "e-").replace("e_p", "e+").strip()
            for p in pieces if p.strip()]


def _number(text: str) -> complex:
    """Evaluate a MATLAB numeric coefficient (``2e-5``, ``-(1+1.5i)``)."""
    t = text.strip().rstrip("*").strip()
    if t in ("", "+"):
        return 1.0
    if t == "-":
        return -1.0
    t = re.sub(r"(\d(?:\.\d*)?(?:[eE][+-]?\d+)?)i\b", r"\1j", t)
    t = t.replace("^", "**")
    try:
        val = eval(t, {"__builtins__": {}}, {})  # noqa: S307 -- numbers only
    except Exception as exc:
        raise ValueError(f"Cannot read the coefficient {text!r}.") from exc
    return complex(val) if isinstance(val, complex) else float(val)


def parse_lin_handle(text: str):
    """Parse the linear part.  Returns ``(var_names, coeffs)`` with
    ``coeffs[k]`` the 5-tuple (lap, biharm, triharm, quadharm, quintharm)
    weights of unknown ``k``; each component may only act on its own
    unknown (MATLAB's diagonal linear part)."""
    names, comps = _split_handle(text)
    if len(comps) != len(names):
        raise ValueError(
            "The linear part must have one component per unknown "
            f"({len(names)} unknowns, {len(comps)} components) in {text!r}.")
    coeffs = []
    for k, comp in enumerate(comps):
        row = [0.0] * len(_OPS)
        for term in _terms(comp):
            m = re.match(r"^(.*?)\b(lap|biharm|triharm|quadharm|quintharm)\(\s*(\w+)\s*\)\s*$",
                         term, flags=re.S)
            if not m:
                if term.strip() in ("0", "+0", "-0"):
                    continue
                raise ValueError(
                    f"Unsupported linear term {term!r}: SPINOP2/SPINOP3 "
                    "linear parts are combinations of lap, biharm, triharm, "
                    "quadharm and quintharm.")
            coef = _number(m.group(1))
            if m.group(3) != names[k]:
                raise ValueError(
                    f"Component {k + 1} of the linear part must act on its own "
                    f"unknown '{names[k]}', got {term!r}.")
            row[_OP_INDEX[m.group(2)]] += coef
        coeffs.append(tuple(row))
    return names, coeffs


def parse_nonlin_handle(text: str):
    """Parse the nonlinear part into ``(var_names, [callables])``, one
    vectorised callable of the unknowns' values per component."""
    from chebfunjax.utils.matlab_expr import matlab_expression
    names, comps = _split_handle(text)
    fns = []
    for comp in comps:
        expr = re.sub(r"(\d(?:\.\d*)?(?:[eE][+-]?\d+)?)i\b", r"\1j", comp)
        fns.append(matlab_expression(expr, names))
    return names, fns


def func2str_text(text: str) -> str:
    """MATLAB ``func2str`` normalisation of an anonymous function: no
    blanks (``@(u,v)[...]``)."""
    return re.sub(r"\s+", "", text.strip())
