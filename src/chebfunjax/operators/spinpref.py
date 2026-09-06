"""Preference objects for the SPIN family of PDE solvers (MATLAB
``spinpref``, ``spinpref2``, ``spinpref3``, ``spinprefsphere``; Fable 5).

The chebfunjax solvers take keyword arguments; these classes hold the
same named preferences with MATLAB's defaults and accept the MATLAB
name/value constructor syntax (``SpinPref(Ylim=(0, 1), dataplot="real")``).

Provenance
----------
MATLAB source : @spinpref/spinpref.m, @spinpref2/spinpref2.m,
    @spinpref3/spinpref3.m, @spinprefsphere/spinprefsphere.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and The
    Chebfun Developers.
"""

from __future__ import annotations


class _SpinPrefBase:
    _defaults: dict = {}

    def __init__(self, **kwargs):
        for k, v in self._defaults.items():
            setattr(self, k, v)
        lower = {k.lower(): k for k in self._defaults}
        for k, v in kwargs.items():
            key = lower.get(k.lower())
            if key is None:
                raise ValueError(f"{type(self).__name__}: unknown preference "
                                 f"{k!r}")
            setattr(self, key, v)

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={getattr(self, k)!r}" for k in self._defaults)
        return f"{type(self).__name__}({body})"


class SpinPref(_SpinPrefBase):
    """Preferences for ``spin`` (1D)."""
    _defaults = dict(dataplot="real", dealias="off", iterplot=20, M=64,
                     Nplot=1024, plot="movie", scheme="etdrk4", Ylim=None)


class SpinPref2(_SpinPrefBase):
    """Preferences for ``spin2`` (2D)."""
    _defaults = dict(Clim=None, colormap="parula", dataplot="real",
                     dealias="off", iterplot=1, M=32, Nplot=128,
                     plot="movie", scheme="etdrk4", view=(0, 90))


class SpinPref3(_SpinPrefBase):
    """Preferences for ``spin3`` (3D)."""
    _defaults = dict(Clim=None, colormap="parula", dataplot="real",
                     dealias="off", iterplot=1, M=32, Nplot=64,
                     plot="movie", scheme="etdrk4", slices=None,
                     view=(-37.5, 30))


class SpinPrefSphere(_SpinPrefBase):
    """Preferences for ``spinsphere``."""
    _defaults = dict(Clim=None, colormap="parula", dataplot="real",
                     dealias="off", grid="off", iterplot=1, Nplot=128,
                     plot="movie", scheme="lirk4", view=(-37.5, 30))


def spinpref(**kwargs) -> SpinPref:
    return SpinPref(**kwargs)


def spinpref2(**kwargs) -> SpinPref2:
    return SpinPref2(**kwargs)


def spinpref3(**kwargs) -> SpinPref3:
    return SpinPref3(**kwargs)


def spinprefsphere(**kwargs) -> SpinPrefSphere:
    return SpinPrefSphere(**kwargs)
