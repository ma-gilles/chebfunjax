"""Global preferences for chebfunjax.

Provides a thread-safe, context-manager-aware preferences object that mirrors
the role of MATLAB Chebfun's ``chebfunpref`` / ``chebpref`` classes.

Usage::

    from chebfunjax.pref import pref

    # Read defaults
    pref.eps          # 2.220446049250313e-16 (machine epsilon)
    pref.max_length   # 65537
    pref.tech         # "chebtech2"
    pref.domain       # (-1.0, 1.0)

    # Modify
    pref.max_length = 1000

    # Temporary override via context manager (thread-safe)
    with pref.context(eps=1e-10):
        ...  # pref.eps == 1e-10 inside this block
    # pref.eps is restored here

    # Reset everything
    pref.reset()

Translated from MATLAB Chebfun classes @chebpref and @chebfunpref (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Default values — matching MATLAB Chebfun factory defaults
# ---------------------------------------------------------------------------

_FACTORY_DEFAULTS: dict[str, Any] = {
    "eps": 2.220446049250313e-16,  # == float64 machine epsilon
    "max_length": 65537,  # max polynomial degree + 1  (2^16 + 1)
    "tech": "chebtech2",  # default representation technology
    "domain": (-1.0, 1.0),  # default interval
    "chop_tol": None,  # if None, adaptive chop uses ``eps``
}


# ---------------------------------------------------------------------------
# ChebPreferences
# ---------------------------------------------------------------------------


class ChebPreferences:
    """Global preferences for chebfunjax.

    Attributes mirror MATLAB Chebfun's ``chebfunpref`` factory defaults:

    * ``eps`` -- construction tolerance (machine epsilon by default).
    * ``max_length`` -- maximum representation length (65537).
    * ``tech`` -- representation technology name (``"chebtech2"``).
    * ``domain`` -- default domain as a 2-tuple (``(-1.0, 1.0)``).
    * ``chop_tol`` -- tolerance for coefficient chopping; ``None`` means use ``eps``.

    The class uses :mod:`contextvars` so that :meth:`context` overrides are
    automatically scoped per-thread / per-async-task.

    Provenance
    ----------
    MATLAB source : @chebpref/chebpref.m, @chebfunpref/chebfunpref.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    chebfunjax.tech.chebtech2
    """

    # The ContextVar holds the current preferences dict.  When no context
    # override is active it falls back to the *module-level* ``_global_prefs``
    # dict, which is shared (and protected by a lock for mutation).
    _cv: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
        "chebfunjax_prefs", default=None
    )
    _lock: threading.Lock = threading.Lock()
    _global_prefs: dict[str, Any] = dict(_FACTORY_DEFAULTS)

    # Keep a copy of factory defaults for reset().
    _factory: dict[str, Any] = dict(_FACTORY_DEFAULTS)

    # Valid preference names (for fast validation).
    _valid_keys: frozenset[str] = frozenset(_FACTORY_DEFAULTS)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current(self) -> dict[str, Any]:
        """Return the active preferences dict (context-local or global)."""
        ctx = self._cv.get()
        if ctx is not None:
            return ctx
        return self._global_prefs

    @staticmethod
    def _validate_key(name: str) -> None:
        if name not in _FACTORY_DEFAULTS:
            raise AttributeError(
                f"'{name}' is not a valid chebfunjax preference. "
                f"Valid preferences: {sorted(_FACTORY_DEFAULTS)}"
            )

    # ------------------------------------------------------------------
    # Attribute access
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Avoid infinite recursion for dunder / private attrs looked up before
        # the class is fully initialised.
        if name.startswith("_"):
            raise AttributeError(name)
        self._validate_key(name)
        return self._current()[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            # Allow setting private/class attrs normally (for __init__, etc.)
            super().__setattr__(name, value)
            return
        self._validate_key(name)
        cur = self._cv.get()
        if cur is not None:
            # We are inside a context() block -- mutate the local copy.
            cur[name] = value
        else:
            with self._lock:
                self._global_prefs[name] = value

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, *names: str) -> None:
        """Reset preferences to factory defaults.

        Parameters
        ----------
        *names : str, optional
            If given, only reset these preferences.  If omitted, reset all.

        Examples
        --------
        >>> pref.max_length = 100
        >>> pref.reset("max_length")
        >>> pref.max_length
        65537
        >>> pref.reset()  # resets everything
        """
        keys = names if names else tuple(self._factory)
        for k in keys:
            self._validate_key(k)
            setattr(self, k, self._factory[k])

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def context(self, **overrides: Any) -> Iterator[ChebPreferences]:
        """Temporarily override preferences.

        Any keyword arguments are applied as preference overrides for the
        duration of the ``with`` block.  On exit the previous values are
        restored.  Overrides are thread-safe (scoped via :mod:`contextvars`).

        Parameters
        ----------
        **overrides
            Preference name/value pairs.  Unknown names raise ``AttributeError``.

        Yields
        ------
        ChebPreferences
            The preferences object (``self``).

        Examples
        --------
        >>> with pref.context(eps=1e-10, max_length=1000):
        ...     print(pref.eps)        # 1e-10
        ...     print(pref.max_length) # 1000
        >>> pref.eps                    # back to machine epsilon
        2.220446049250313e-16
        """
        for k in overrides:
            self._validate_key(k)

        # Snapshot current state into a new context-local dict.
        snapshot = dict(self._current())
        snapshot.update(overrides)
        token = self._cv.set(snapshot)
        try:
            yield self
        finally:
            self._cv.reset(token)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cur = self._current()
        lines = ["ChebPreferences("]
        for k in sorted(cur):
            lines.append(f"    {k}={cur[k]!r},")
        lines.append(")")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict of the current preference values."""
        return dict(self._current())


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

pref = ChebPreferences()
"""Module-level preferences singleton.

Import and use directly::

    from chebfunjax.pref import pref
    pref.eps
"""
