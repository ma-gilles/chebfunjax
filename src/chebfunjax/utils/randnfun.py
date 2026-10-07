"""Compatibility interface: the historical periodic JAX-key convenience API.

The default wavelength .2 and implicit periodic option are retained here only.
Use chebfunjax.randnfun for MATLAB defaults (lambda1, nonperiodic, [-1,1]).
All three interfaces share construction equations and normalization.

Provenance
----------
MATLAB source : randnfun.m (explicit trig option)
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
from __future__ import annotations


def randnfun(lam=0.2, domain=(-1.0, 1.0), *, key=None, seed=None,
             big=False, cmplx=False):
    """Periodic compatibility wrapper; no fixed default realization.

    Provenance
    ----------
    MATLAB source : randnfun.m (explicit trig option)
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    from chebfunjax.utils._randnfun import randnfun as source_randnfun
    return source_randnfun(lam, domain, 'trig', key=key, seed=seed,
                          big=big, cmplx=cmplx)
