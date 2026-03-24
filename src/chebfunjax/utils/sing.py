# uses-numpy: singularity detection uses numpy-based finite-difference sampling (not JIT-safe)
"""Singularity detection at function endpoints.

Detects algebraic (pole or branch-point) singularities at ``x = ±1`` by
sampling the function close to the endpoint and estimating the blow-up
exponent via a log-ratio method.

Translated from MATLAB Chebfun ``@singfun/findPoleOrder.m``,
``@singfun/findSingOrder.m``, ``@singfun/findSingExponents.m``
(commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

Provenance
----------
MATLAB source : @singfun/findPoleOrder.m, @singfun/findSingOrder.m,
               @singfun/findSingExponents.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = [
    "find_pole_order",
    "find_sing_order",
    "find_sing_exponents",
]

# Default maximum pole order to test before giving up.
_MAX_POLE_ORDER: int = 20

# Ratio threshold used in the integer pole-order finder.
_TEST_RATIO: float = 1.01


def find_pole_order(
    op: Callable,
    endpoint: str,
    *,
    max_pole_order: int = _MAX_POLE_ORDER,
) -> int:
    """Find the integer order of a pole at ``x = -1`` or ``x = 1``.

    Samples the function at ``10^{-1}, 10^{-2}, ..., 10^{-15}`` away from
    the specified endpoint and finds the smallest non-negative integer *k*
    such that ``op(x) * dist^k`` is bounded as ``dist -> 0``.

    Parameters
    ----------
    op : callable
        Function handle with signature ``op(x) -> float-like``.
    endpoint : {'left', 'right'}
        ``'right'`` tests near ``x = 1``; ``'left'`` tests near ``x = -1``.
    max_pole_order : int, default 20
        Maximum integer pole order to test before raising.

    Returns
    -------
    pole_order : int
        Negative integer exponent (e.g. ``-2`` for a double pole).
        Returns ``0`` if no pole is detected.

    Raises
    ------
    ValueError
        If ``endpoint`` is not ``'left'`` or ``'right'``.
    RuntimeError
        If the pole order exceeds ``max_pole_order``.

    Examples
    --------
    >>> find_pole_order(lambda x: 1 / (1 - x), 'right')
    -1
    >>> find_pole_order(lambda x: 1 / (1 + x)**2, 'left')
    -2
    >>> find_pole_order(lambda x: x**2, 'right')
    0

    Provenance
    ----------
    MATLAB source : @singfun/findPoleOrder.m
    Chebfun commit: 7574c77
    """
    endpoint = endpoint.lower()
    if endpoint not in ("left", "right"):
        raise ValueError(
            f"endpoint must be 'left' or 'right', got {endpoint!r}"
        )

    x_dist = 10.0 ** np.arange(-1, -16, -1)  # distances from endpoint

    if endpoint == "right":
        fvals = np.abs(np.array([float(op(1.0 - dx)) for dx in x_dist], dtype=np.float64))
    else:
        fvals = np.abs(np.array([float(op(-1.0 + dx)) for dx in x_dist], dtype=np.float64))

    # Remove any ±inf values (they occur exactly AT the pole — skip)
    valid = np.isfinite(fvals)
    x_dist = x_dist[valid]
    fvals = fvals[valid]

    if len(fvals) < 2:
        return 0

    pole_order = 0
    smooth_vals = fvals.copy()

    # Multiply by x_dist successively until values no longer grow toward endpoint
    while (
        np.all(smooth_vals[1:] / smooth_vals[:-1] > _TEST_RATIO)
        and pole_order <= max_pole_order
    ):
        pole_order += 1
        smooth_vals = smooth_vals * x_dist

    if pole_order > max_pole_order:
        raise RuntimeError(
            f"Pole order detection failed: exceeded max_pole_order={max_pole_order}."
        )

    # Return as a *negative* exponent (pole of order k => exponent -k)
    return -pole_order


def find_sing_order(
    op: Callable,
    endpoint: str,
    *,
    max_pole_order: int = _MAX_POLE_ORDER,
) -> float:
    """Find the order of a (possibly fractional) singularity at an endpoint.

    Uses a log-ratio estimate of the algebraic blow-up/decay exponent,
    bounded above by the result of :func:`find_pole_order`.

    Parameters
    ----------
    op : callable
        Function handle with signature ``op(x) -> float-like``.
    endpoint : {'left', 'right'}
        ``'right'`` tests near ``x = 1``; ``'left'`` tests near ``x = -1``.
    max_pole_order : int, default 20
        Passed to :func:`find_pole_order`.

    Returns
    -------
    sing_order : float
        Negative exponent describing the singularity (e.g. ``-1.5``
        for ``(1-x)^{-1.5}``).  Returns ``0.0`` if no singularity is
        detected or if the estimated exponent is ``>= 1`` (branch points of
        order >= 1 converge well without singular treatment).

    Examples
    --------
    >>> import numpy as np
    >>> find_sing_order(lambda x: (1 - x)**(-1.5), 'right')  # doctest: +SKIP
    -1.5
    >>> find_sing_order(lambda x: x**2, 'right')
    0.0

    Provenance
    ----------
    MATLAB source : @singfun/findSingOrder.m
    Chebfun commit: 7574c77
    """
    endpoint = endpoint.lower()
    if endpoint not in ("left", "right"):
        raise ValueError(
            f"endpoint must be 'left' or 'right', got {endpoint!r}"
        )

    # Get an integer upper bound on the order first
    pole_bound = -find_pole_order(op, endpoint, max_pole_order=max_pole_order)

    # Sample close to the endpoint: eps * (11, 10, ..., 2)
    eps = np.finfo(np.float64).eps
    x_dist = eps * np.arange(11, 1, -1, dtype=np.float64)

    if endpoint == "right":
        fvals = np.array([float(op(1.0 - dx)) for dx in x_dist], dtype=np.float64)
    else:
        fvals = np.array([float(op(-1.0 + dx)) for dx in x_dist], dtype=np.float64)

    fvals = np.abs(fvals)

    # Estimate exponent via log-ratio of adjacent samples
    valid = (fvals > 0) & np.isfinite(fvals) & (x_dist > 0)
    if np.sum(valid) < 2:
        return 0.0

    fv = fvals[valid]
    xd = x_dist[valid]

    # f(x) ~ dist^exponent near the endpoint.
    # log(|f(x_i)| / |f(x_{i+1})|) / log(x_dist_i / x_dist_{i+1}) = exponent.
    # For blow-up (pole/sing): fv increases as dist decreases => ratio < 0.
    # For a root: fv decreases => ratio > 0.
    log_fv = np.log(fv[:-1] / fv[1:])
    log_xd = np.log(xd[:-1] / xd[1:])

    finite = np.isfinite(log_fv) & np.isfinite(log_xd) & (log_xd != 0)
    if not np.any(finite):
        return 0.0

    # exponent < 0 means blow-up; take median for robustness
    exponent = float(np.median(log_fv[finite] / log_xd[finite]))

    # Clip magnitude to the integer pole-order bound (pole_bound >= 1)
    if exponent < 0:
        exponent = max(exponent, -float(pole_bound))

    # MATLAB: discard positive exponents (roots of order >= 1 are fine)
    if exponent >= 1.0:
        return 0.0

    return exponent


def find_sing_exponents(
    op: Callable,
    sing_type: tuple[str, str] = ("pole", "pole"),
    *,
    max_pole_order: int = _MAX_POLE_ORDER,
) -> tuple[float, float]:
    """Detect algebraic singularity exponents at both endpoints of ``[-1, 1]``.

    Wrapper combining :func:`find_pole_order` and :func:`find_sing_order`
    for each endpoint, dispatching on ``sing_type``.

    Parameters
    ----------
    op : callable
        Function handle with signature ``op(x) -> float-like``, defined on
        the reference interval ``[-1, 1]``.
    sing_type : tuple[str, str], default ``('pole', 'pole')``
        Pair of strings for the left and right endpoints respectively.
        Each element must be one of:

        ``'pole'``
            Integer-order pole; uses :func:`find_pole_order`.
        ``'sing'``
            Fractional-order (branch-point) singularity; uses
            :func:`find_sing_order`.
        ``'none'``
            No singularity at this endpoint; returns exponent ``0.0``.

    max_pole_order : int, default 20
        Passed to the underlying detection functions.

    Returns
    -------
    exponents : tuple[float, float]
        ``(left_exponent, right_exponent)`` as floating-point numbers.
        Negative values indicate blow-up; zero indicates smoothness.

    Examples
    --------
    >>> find_sing_exponents(lambda x: 1 / (1 + x), ('pole', 'none'))
    (-1, 0.0)

    Provenance
    ----------
    MATLAB source : @singfun/findSingExponents.m
    Chebfun commit: 7574c77
    """
    endpoints = ("left", "right")
    result = []

    for end, stype in zip(endpoints, sing_type):
        stype_lower = stype.lower()
        if stype_lower == "pole":
            exp = float(find_pole_order(op, end, max_pole_order=max_pole_order))
        elif stype_lower in ("sing", "root"):
            exp = find_sing_order(op, end, max_pole_order=max_pole_order)
        elif stype_lower == "none":
            exp = 0.0
        else:
            raise ValueError(
                f"Unknown sing_type element {stype!r}. "
                "Expected 'pole', 'sing', 'root', or 'none'."
            )
        result.append(exp)

    return (result[0], result[1])
