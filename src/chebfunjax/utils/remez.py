"""Deprecated native REMEZ public output adapter, Chebfun7574c77."""

import warnings

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.utils.minimax import minimax


def remez(f, m, n=None, *, tol=None, max_iter=None, silent=False):
    """Return polynomial P or rational (P,Q) through public MINIMAX.

    Python uses a Chebfun return for the polynomial call and a pair for the
    rational call, matching the source page's requested output counts.
    The native deprecation warning is emitted even when silent=True.

    Provenance
    ----------
    MATLAB source : @chebfun/remez.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    warnings.warn(
        "CHEBFUN:CHEBFUN:remez: This command is deprecated. Use minimax instead.",
        UserWarning,
        stacklevel=2,
    )
    if not isinstance(f, Chebfun):
        raise TypeError("remez expects a Chebfun")
    options = dict(
        domain=(float(f.domain.a), float(f.domain.b)), tol=tol, max_iter=max_iter, silent=silent
    )
    if n is None:
        result = minimax(f, m, **options)
        return Chebfun.from_coeffs(result.coeffs, result.domain)
    result = minimax(f, m, rational=True, denom=n, **options)
    return result.as_chebfuns()
