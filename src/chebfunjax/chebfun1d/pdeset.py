"""PDE and ODE option factory adapted from Chebfun ``pdeset``/MATLAB ``odeset``.

The factory stores empty values as ``None`` and defers value validation to the
consumer, as the MATLAB source does.  Display is explicit because Python has
no equivalent of MATLAB's ``nargout``.

Provenance
----------
Chebfun ``pdeset.m`` and MATLAB R2025b ``odeset.m``; Chebfun commit 7574c77.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TextIO

ODE_FIELDS = (
    "AbsTol",
    "BDF",
    "Events",
    "InitialStep",
    "Jacobian",
    "JConstant",
    "JPattern",
    "Mass",
    "MassSingular",
    "MaxOrder",
    "MaxStep",
    "MinStep",
    "NonNegative",
    "NormControl",
    "OutputFcn",
    "OutputSel",
    "Refine",
    "RelTol",
    "Stats",
    "Vectorized",
    "MStateDependence",
    "MvPattern",
    "InitialSlope",
)

PDE_FIELDS = (
    "Eps",
    "N",
    "Plot",
    "HoldPlot",
    "YLim",
    "PlotStyle",
    "PDEflag",
    "AdjustBCs",
    "ODESolver",
)

_ODE_BY_LOWER = {name.lower(): name for name in ODE_FIELDS}
_PDE_BY_LOWER = {name.lower(): name for name in PDE_FIELDS}

_ODE_DISPLAY = (
    "          AbsTol: [ positive scalar or vector {1e-6} ]",
    "          RelTol: [ positive scalar {1e-3} ]",
    "     NormControl: [ on | {off} ]",
    "     NonNegative: [ vector of integers ]",
    "       OutputFcn: [ function_handle ]",
    "       OutputSel: [ vector of integers ]",
    "          Refine: [ positive integer ]",
    "           Stats: [ on | {off} ]",
    "     InitialStep: [ positive scalar ]",
    "         MaxStep: [ positive scalar ]",
    "         MinStep: [ positive scalar ]",
    "             BDF: [ on | {off} ]",
    "        MaxOrder: [ 1 | 2 | 3 | 4 | {5} ]",
    "        Jacobian: [ matrix | function_handle ]",
    "        JPattern: [ sparse matrix ]",
    "      Vectorized: [ on | {off} ]",
    "            Mass: [ matrix | function_handle ]",
    "MStateDependence: [ none | {weak} | strong ]",
    "       MvPattern: [ sparse matrix ]",
    "    MassSingular: [ yes | no | {maybe} ]",
    "    InitialSlope: [ vector ]",
    "          Events: [ function_handle ]",
)

_PDE_DISPLAY = (
    "             Eps: [ positive scalar {1e-6} ]",
    "               N: [ {NaN} | positive integer  ]",
    "            Plot: [ on | {off} ]",
    "        HoldPlot: [ on | {off} ]",
    "            YLim: [ 2x1 vector | {NaN} ]",
    "       PlotStyle: [ string | '-']",
    "         PDEflag: [ vector of logicals {true} ]",
    "       AdjustBCs: [ logical {true} ]",
    "       ODESolver: [ {@ode15s} | @ode23t ]",
)


class PdeSetError(ValueError):
    """A pdeset/odeset parse error with its source error identifier."""

    def __init__(self, identifier: str, message: str):
        super().__init__(f"{identifier}: {message}")
        self.identifier = identifier


def _empty(value) -> bool:
    """MATLAB ``isempty`` for the ordinary Python option-value adapters."""
    if value is None:
        return True
    if isinstance(value, (str, bytes, list, tuple)):
        return len(value) == 0
    size = getattr(value, "size", None)
    return isinstance(size, int) and size == 0


def _as_matlab_logical(value) -> bool:
    """Convert a scalar/vector Python value using MATLAB ``logical``/``if``."""
    if _empty(value):
        return False
    if isinstance(value, (list, tuple)):
        # MATLAB evaluates logical(in) for the whole array before if.
        converted = [_as_matlab_logical(item) for item in value]
        return all(converted)
    tolist = getattr(value, "tolist", None)
    if callable(tolist) and not isinstance(value, (str, bytes)):
        converted = tolist()
        if converted is not value:
            return _as_matlab_logical(converted)
    if isinstance(value, complex):
        raise PdeSetError(
            "MATLAB:nologicalcomplex", "complex values cannot be converted to logical"
        )
    try:
        is_nan = math.isnan(value)
    except (TypeError, ValueError):
        is_nan = False
    if is_nan:
        raise PdeSetError(
            "MATLAB:nologicalnan", "NaN values cannot be converted to logical"
        )
    try:
        return bool(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("option value cannot be converted to MATLAB logical") from exc


def _pde_value(name: str, value):
    if name in ("Plot", "HoldPlot") and not isinstance(value, str):
        return "on" if _as_matlab_logical(value) else "off"
    return value


def _parse_error(identifier: str, detail: str):
    raise PdeSetError(identifier, detail)


def pdeset(*args) -> dict[str, object]:
    """Create or update a 32-field PDE/ODE options mapping.

    ``pdeset(old, name, value, ...)`` accepts an existing mapping first.
    Names are case-insensitive but must be complete canonical names, matching
    Chebfun's pdeset pre-parser.  Every unspecified field is ``None`` (MATLAB
    ``[]``).  The factory intentionally does not validate option values.

    MATLAB ``pdeset`` displays help when called with no output arguments.
    Python cannot inspect the caller's output count, so call
    :func:`show_pdeset_options` for that display.

    Provenance
    ----------
    Native sources: Chebfun ``pdeset.m`` and MATLAB R2025b ``odeset.m``.
    Original source copyrights: 2017 The University of Oxford and Chebfun
    developers; 1984-2024 The MathWorks, Inc.
    Chebfun commit: 7574c77
    """
    remaining = list(args)
    old: Mapping | None = None
    if remaining and isinstance(remaining[0], Mapping):
        old = remaining.pop(0)

    options = {name: None for name in ODE_FIELDS}
    pde_values = {name: None for name in PDE_FIELDS}

    if old is not None:
        # odeset merges only canonical-case recognized field names and ignores
        # empty values in leading option structures.
        for name in ODE_FIELDS:
            if name in old and not _empty(old[name]):
                options[name] = old[name]
        # pdeset uses isfield with canonical spelling when reading old PDE
        # fields, then attaches them after odeset has rebuilt its fields.
        for name in PDE_FIELDS:
            if name in old:
                pde_values[name] = old[name]

    ode_pairs: list[tuple[str, object]] = []
    pde_pairs: list[tuple[str, object]] = []
    i = 0
    while i < len(remaining) - 1:
        name, value = remaining[i], remaining[i + 1]
        if not isinstance(name, str):
            _parse_error("MATLAB:odeset:NoPropName", f"option name at position {i + 1} is not text")
        lower = name.lower()
        if lower in _ODE_BY_LOWER:
            ode_pairs.append((_ODE_BY_LOWER[lower], value))
        else:
            # onoff runs during the native pre-parser, before odeset diagnoses
            # odd arguments and before unknown PDE names are attached.
            canonical = _PDE_BY_LOWER.get(lower, "")
            pde_pairs.append((name, _pde_value(canonical, value)))
        i += 2

    # pdeset leaves a trailing unmatched argument for odeset to diagnose.
    if i < len(remaining):
        _parse_error("MATLAB:odeset:ArgNameValueMismatch", "name/value arguments must be paired")

    # R2025b odeset's leading-struct merge keeps nonempty values.  Name/value
    # values then overwrite them left-to-right, including explicit empties.
    for name, value in ode_pairs:
        options[name] = value

    for supplied_name, value in pde_pairs:
        name = _PDE_BY_LOWER.get(supplied_name.lower())
        if name is None:
            _parse_error(
                "CHEBFUN:pdeset:unknownOption",
                f"Unrecognized property name {supplied_name}.",
            )
        pde_values[name] = value

    options.update(pde_values)
    return options


def show_pdeset_options(file: TextIO | None = None) -> None:
    """Print the ``odeset`` then ``pdeset`` option descriptions in source order.

    Provenance
    ----------
    Native sources: MATLAB R2025b ``odeset.m`` and Chebfun ``pdeset.m``.
    Original source copyrights: 1984-2024 The MathWorks, Inc.; 2017 The
    University of Oxford and Chebfun developers.
    Chebfun commit: 7574c77
    """
    import sys

    stream = sys.stdout if file is None else file
    for line in (*_ODE_DISPLAY, "", *_PDE_DISPLAY):
        print(line, file=stream)


__all__ = ["ODE_FIELDS", "PDE_FIELDS", "PdeSetError", "pdeset", "show_pdeset_options"]
