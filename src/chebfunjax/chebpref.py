"""MATLAB-shaped preference objects: chebfunpref / cheboppref.

The separate ContextVar preference adapter lives in :mod:`chebfunjax.pref`.
This module supplies the MATLAB preference object consulted by source ports
(struct construction, techPrefs merging and passthrough, mergeTechPrefs,
setDefaults / factory reset) so preference-manipulating code and the
MATLAB test suite translate directly.

Provenance
----------
MATLAB source : @chebfunpref/chebfunpref.m, @cheboppref/cheboppref.m,
    chebpref.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

import copy
import threading
import warnings

_EPS = 2.220446049250313e-16


class DotDict(dict):
    """A dict with attribute access, used for techPrefs substructures."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    @staticmethod
    def wrap(d):
        out = DotDict()
        for k, v in d.items():
            out[k] = DotDict.wrap(v) if isinstance(v, dict) else v
        return out


def _factory_tech_prefs(tech="chebtech2") -> DotDict:
    # @chebtech/techPref.m59-67, @trigtech/techPref.m63-71 (7574c77).
    # Public constructor aliases are supported; unknown technologies never
    # inherit polynomial defaults silently. An explicit provider is allowed.
    provider = getattr(tech, "techPref", None)
    if callable(provider):
        supplied = provider()
        if not isinstance(supplied, dict):
            raise TypeError("Tech techPref() must return a mapping dict")
        if any(isinstance(value, dict) for value in supplied.values()):
            raise NotImplementedError("Nested Tech factory defaults require path-aware mutation")
        return DotDict.wrap(copy.deepcopy(supplied))
    key = (tech.__name__ if isinstance(tech, type) else str(tech)).lower().lstrip("@")
    if key not in {"chebtech", "chebtech1", "chebtech2", "trigtech", "trig", "periodic"}:
        raise ValueError(f"Unsupported preference technology {tech!r}")
    periodic = key in {"trigtech", "trig", "periodic"}
    out = DotDict.wrap({
        "chebfuneps": _EPS,
        "maxLength": 65536 if periodic else 65537,
        "minSamples": 17,
        "fixedLength": None,
        "extrapolate": False,
        "sampleTest": True,
        "refinementFunction": "nested",
        "happinessCheck": "standard",
    })
    out["gridType" if periodic else "useTurbo"] = 2 if periodic else False
    return out


def _factory_top() -> dict:
    return {
        "domain": (-1.0, 1.0),
        "splitting": False,
        "splitPrefs": DotDict.wrap({"splitLength": 160,
                                    "splitMaxLength": 6000}),
        "blowup": False,
        "blowupPrefs": DotDict.wrap({"exponentTol": 1.1e-11,
                                     "maxPoleOrder": 20,
                                     "defaultSingType": "sing"}),
        "enableDeltaFunctions": True,
        "enableFunqui": False,  # @chebfunpref/chebfunpref.m731, pin7574c77
        "deltaPrefs": DotDict.wrap({"deltaTol": 1e-9,
                                    "proximityTol": 1e-11}),
        "tech": "chebtech2",
        "cheb2Prefs": DotDict.wrap({"chebfun2eps": _EPS,
                                    "maxRank": 513,
                                    "sampleTest": True}),
        "cheb3Prefs": DotDict.wrap({"chebfun3eps": _EPS,
                                    "maxRank": 128,
                                    "sampleTest": True}),
    }


_MISSING = object()


class _TechPrefView(DotDict):
    """Resolved dict view; mutations record only explicit raw overrides.

    Python deletion removes raw overrides: inherited-only keys raise KeyError.
    Copies materialize a detached resolved DotDict. Native source reads/writes:
    @chebfunpref/chebfunpref.m342-408, pin7574c77. Deletion is a Python adapter.
    """

    def __init__(self, owner):
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_dirty", True)

    def _ensure(self):
        if self._dirty:
            self._refresh()

    def _refresh(self):
        resolved = _factory_tech_prefs(self._owner._top["tech"])
        resolved.update(self._owner._tech_overrides)
        dict.clear(self)
        dict.update(self, resolved)
        object.__setattr__(self, "_dirty", False)

    def __getitem__(self, key):
        self._ensure()
        return dict.__getitem__(self, key)

    def __iter__(self):
        self._ensure()
        return dict.__iter__(self)

    def __len__(self):
        self._ensure()
        return dict.__len__(self)

    def __contains__(self, key):
        self._ensure()
        return dict.__contains__(self, key)

    def get(self, key, default=None):
        self._ensure()
        return dict.get(self, key, default)

    def keys(self):
        self._ensure()
        return dict.keys(self)

    def items(self):
        self._ensure()
        return dict.items(self)

    def values(self):
        self._ensure()
        return dict.values(self)

    def __eq__(self, other):
        self._ensure()
        if isinstance(other, _TechPrefView):
            other._ensure()
        return dict.__eq__(self, other)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        self._ensure()
        return dict.__repr__(self)

    def __setitem__(self, key, value):
        value = DotDict.wrap(value) if isinstance(value, dict) else value
        self._owner._tech_overrides[key] = value
        object.__setattr__(self, "_dirty", True)

    def __delitem__(self, key):
        del self._owner._tech_overrides[key]
        object.__setattr__(self, "_dirty", True)

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, key, default=_MISSING):
        raw = self._owner._tech_overrides
        if key not in raw:
            if default is _MISSING:
                raise KeyError(key)
            return default
        value = raw[key]
        del self[key]
        return value

    def popitem(self):
        raw = self._owner._tech_overrides
        if not raw:
            raise KeyError("No explicit technology overrides")
        key = next(reversed(raw))
        return key, self.pop(key)

    def clear(self):
        self._owner._tech_overrides.clear()
        object.__setattr__(self, "_dirty", True)

    def __or__(self, other):
        if not isinstance(other, dict):
            return NotImplemented
        self._ensure()
        return dict(self) | dict(other)

    def __ror__(self, other):
        if not isinstance(other, dict):
            return NotImplemented
        self._ensure()
        return dict(other) | dict(self)

    def __reversed__(self):
        self._ensure()
        return dict.__reversed__(self)

    def __ior__(self, other):
        self.update(other)
        return self

    def copy(self):
        return DotDict(dict(self))

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return DotDict.wrap(copy.deepcopy(dict(self), memo))


class ChebfunPref:
    """Native raw overrides with lazy selected-Tech default resolution.

    Provenance: @chebfunpref/chebfunpref.m264-408,532-733, pin7574c77.
    Reads never record defaults as explicit; object copies preserve omission.
    Public dict/view copies and mergeTechPrefs intentionally materialize values.
    Unknown-Tech-pref warning parity remains open; supplied fields are retained.
    """

    _defaults: "ChebfunPref | None" = None

    def __init__(self, src=None, overrides=_MISSING, **kwargs):
        if overrides is not _MISSING and not isinstance(src, ChebfunPref):
            raise TypeError("Two-input ChebfunPref requires an object base")
        if src is not None and not isinstance(src, (ChebfunPref, dict)):
            raise TypeError("ChebfunPref input must be an object or dict")
        base = src if isinstance(src, ChebfunPref) else type(self).__dict__.get("_defaults")
        object.__setattr__(self, "_top", copy.deepcopy(base._top) if base is not None else _factory_top())
        object.__setattr__(self, "_tech_overrides", copy.deepcopy(base._tech_overrides) if base is not None else DotDict())
        object.__setattr__(self, "_tech_view", None)
        if isinstance(src, dict):
            self._absorb(src)
        if overrides is not _MISSING:
            if isinstance(overrides, ChebfunPref):
                supplied = copy.deepcopy(overrides._top)
                supplied["techPrefs"] = copy.deepcopy(overrides.techPrefs)
            elif isinstance(overrides, dict):
                supplied = overrides
            else:
                raise TypeError("Second ChebfunPref input must be an object or dict")
            self._absorb(supplied)
        self._absorb(kwargs)

    @property
    def techPrefs(self):
        if self._tech_view is None:
            object.__setattr__(self, "_tech_view", _TechPrefView(self))
        # Native subsref resolves the selected provider on every owner read.
        # Defer until values are read so Python nested assignments stay raw.
        object.__setattr__(self._tech_view, "_dirty", True)
        return self._tech_view

    def _refresh_view(self):
        # Raw writes must not call a possibly unsupported/custom provider.
        if self._tech_view is not None:
            object.__setattr__(self._tech_view, "_dirty", True)

    def _absorb(self, d):
        # Source constructor merges substructures one level, not recursively.
        for key, value in d.items():
            if key == "techPrefs":
                if not isinstance(value, dict):
                    raise TypeError("techPrefs must be a dict")
                self._tech_overrides.update(DotDict.wrap(copy.deepcopy(value)))
            elif key in self._top:
                if isinstance(self._top[key], dict) and isinstance(value, dict):
                    self._top[key].update(DotDict.wrap(copy.deepcopy(value)))
                else:
                    self._top[key] = copy.deepcopy(value)
            else:
                self._tech_overrides[key] = DotDict.wrap(copy.deepcopy(value)) if isinstance(value, dict) else copy.deepcopy(value)
        self._refresh_view()

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._top:
            return self._top[name]
        try:
            return self.techPrefs[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        elif name == "techPrefs":
            # Augmented assignment has already updated this view; do not
            # materialize inherited defaults in its implicit writeback.
            if value is self._tech_view:
                return
            if not isinstance(value, dict):
                raise TypeError("techPrefs must be a dict")
            object.__setattr__(self, "_tech_overrides", DotDict.wrap(copy.deepcopy(dict(value))))
            self._refresh_view()
        elif name in self._top:
            self._top[name] = DotDict.wrap(value) if isinstance(value, dict) else value
            if name == "tech":
                self._refresh_view()
        else:
            self._tech_overrides[name] = DotDict.wrap(value) if isinstance(value, dict) else value
            self._refresh_view()

    def __copy__(self):
        return self.__deepcopy__({})

    def __deepcopy__(self, memo):
        out = type(self).__new__(type(self))
        memo[id(self)] = out
        object.__setattr__(out, "_top", copy.deepcopy(self._top, memo))
        object.__setattr__(out, "_tech_overrides", copy.deepcopy(self._tech_overrides, memo))
        object.__setattr__(out, "_tech_view", None)
        return out

    def __eq__(self, other):
        return (isinstance(other, ChebfunPref) and self._top == other._top
                and self.techPrefs == other.techPrefs)

    __hash__ = None

    @staticmethod
    def mergeTechPrefs(p, q) -> DotDict:
        """Materialize each object's own selected defaults; later fields win.

        Native @chebfunpref/chebfunpref.m532-562, pin7574c77.
        """
        def materialize(x):
            return copy.deepcopy(x.techPrefs) if isinstance(x, ChebfunPref) else DotDict.wrap(copy.deepcopy(dict(x)))
        out = materialize(p)
        out.update(materialize(q))
        return out

    @classmethod
    def getFactoryDefaults(cls):
        # Build without temporarily mutating session state; retain subclass
        # constructor defaults via a supplied factory object, not session copy.
        raw = cls.__new__(cls)
        object.__setattr__(raw, "_top", _factory_top())
        object.__setattr__(raw, "_tech_overrides", DotDict())
        object.__setattr__(raw, "_tech_view", None)
        return cls(raw)

    @classmethod
    def setDefaults(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], str) and args[0] == "factory" and not kwargs:
            cls._defaults = None
            return
        if len(args) == 1 and isinstance(args[0], (ChebfunPref, dict)) and not kwargs:
            cls._defaults = cls(args[0])
            return
        # @chebpref/chebpref.m: no inputs and unpaired arguments error.
        if not args and not kwargs:
            raise TypeError("setDefaults requires at least one argument")
        if len(args) % 2:
            raise TypeError("setDefaults requires name/value pairs")
        # Native manageDefaultPrefs applies pairs in order, including repeats.
        pairs = list(zip(args[0::2], args[1::2]))
        pairs.extend(kwargs.items())
        base = cls()
        factory = cls.getFactoryDefaults()
        for key, value in pairs:
            want_factory = isinstance(value, str) and value == "factory"
            if isinstance(key, (list, tuple)):
                # Source two-tier branch tests the stored raw structure,
                # never the resolved technology-default view.
                if len(key) != 2 or not all(isinstance(k, str) for k in key):
                    raise TypeError("Two-tier preference names require two strings")
                parent, child = key
                target = base._tech_overrides if parent == "techPrefs" else base._top.get(parent)
                if not isinstance(target, dict) or child not in target:
                    raise KeyError(tuple(key))
                if want_factory:
                    original = factory._tech_overrides if parent == "techPrefs" else factory._top.get(parent)
                    if not isinstance(original, dict) or child not in original:
                        raise KeyError(tuple(key))
                    value = copy.deepcopy(original[child])
                target[child] = DotDict.wrap(value) if isinstance(value, dict) else value
                base._refresh_view()
            elif want_factory:
                if key == "techPrefs":
                    base.techPrefs = {}
                elif key in factory._top:
                    setattr(base, key, copy.deepcopy(factory._top[key]))
                else:
                    del base._tech_overrides[key]
                    base._refresh_view()
            else:
                setattr(base, key, value)
            # Native persistent manager commits each pair before the next.
            cls._defaults = copy.deepcopy(base)


class ChebopPref(ChebfunPref):
    """MATLAB ``cheboppref``: the chebop preference structure.

    Provenance
    ----------
    MATLAB source : @cheboppref/cheboppref.m
    Chebfun commit: 7574c77
    """

    # Native factory @cheboppref/cheboppref.m459-476. The existing Python
    # selector "standard" represents @standardCheck; arbitrary native handle
    # invocation and entry-specific alias parsing are not implemented here.
    _defaults: "ChebopPref | None" = None

    def __init__(self, src=None, **kwargs):
        # Install subclass fields before absorbing overrides, and start from
        # session defaults as the source constructor does for a struct input.
        super().__init__(src if isinstance(src, ChebfunPref) else None)
        top = self.__dict__["_top"]
        for k, v in (("discretization", "values"),
                     ("bvpTol", 5e-13), ("scale", float("nan")),
                     ("lambdaMin", 1e-6), ("happinessCheck", "standard"),
                     ("minDimension", 32),
                     ("maxDimension", 4096), ("ivpAbsTol", 1e5 * _EPS),
                     ("ivpRelTol", 100 * _EPS), ("ivpRestartSolver", True),
                     ("damping", True),
                     ("maxIter", 25), ("plotting", "off"),
                     ("display", "off"), ("vectorize", True),
                     ("ivpSolver", "ode113")):
            top.setdefault(k, v)
        if isinstance(src, dict):
            self._absorb(src)
        if kwargs:
            self._absorb(kwargs)


# ---------------------------------------------------------------------------
# Deprecated global toggles (MATLAB splitting.m / blowup.m)
# ---------------------------------------------------------------------------

_SPLITTING_WARNING_LOCK = threading.Lock()
_SPLITTING_WARNING_EMITTED = False
_SPLITTING_WARNING_TEXT = (
    "The syntax 'splitting on' is deprecated.\n"
    "Please see CHEBFUNPREF documentation for further details."
)

def splitting(state=None) -> str:
    """Query or set the global splitting preference.

    A setter returns the state from before mutation, like MATLAB when an
    output is requested. A no-argument Python call returns the current state;
    Python cannot infer MATLAB's ``nargout`` to decide whether to print.

    Provenance
    ----------
    MATLAB source : splitting.m
    Chebfun commit: 7574c77
    """
    global _SPLITTING_WARNING_EMITTED
    old = "on" if bool(ChebfunPref().splitting) else "off"
    if state is None:
        return old

    # MATLAB emits warning ID CHEBFUN:splitting:deprecated and disables it
    # before option validation. Python has no warning-ID field, so retain the
    # exact message and process-wide one-time behavior.
    with _SPLITTING_WARNING_LOCK:
        if not _SPLITTING_WARNING_EMITTED:
            warnings.warn(_SPLITTING_WARNING_TEXT, FutureWarning, stacklevel=2)
            # Source warning('off', id) runs only if warning() returned. If
            # Python promotes the warning to an exception, a retry warns again.
            _SPLITTING_WARNING_EMITTED = True

    # Keep the existing Python coercion for unsupported inputs; invalid
    # values still follow the source UnknownOption error after the warning.
    key = str(state).lower()
    if key not in ("on", "off"):
        raise ValueError(
            "CHEBFUN:splitting:UnknownOption: Unknown splitting option: "
            "only ON and OFF are valid options."
        )
    ChebfunPref.setDefaults("splitting", key == "on")
    return old


def blowup(state=None) -> int:
    """Query or set the session default ``blowup`` preference (MATLAB
    ``blowup()``): ``0``/``'off'`` disables singularity detection,
    ``1``/``'on'`` detects poles (``defaultSingType = 'pole'``), ``2``
    detects general branch singularities (``'sing'``).  Returns the
    mode in force BEFORE the call.

    Provenance
    ----------
    MATLAB source : blowup.m
    Chebfun commit: 7574c77
    """
    p = ChebfunPref()
    if not p.blowup:
        old = 0
    else:
        old = 1 if str(p.blowupPrefs.defaultSingType).lower() == "pole" \
            else 2
    if state is not None:
        if isinstance(state, str):
            key = state.lower()
            mode = {"off": 0, "on": 1}.get(key)
            if mode is None:
                raise ValueError("blowup: state must be 'on', 'off', 0, 1 "
                                 "or 2.")
        else:
            mode = int(state)
            if mode not in (0, 1, 2):
                raise ValueError("blowup: mode must be 0, 1 or 2.")
        q = ChebfunPref()
        q.blowup = mode != 0
        if mode == 1:
            q.blowupPrefs.defaultSingType = "pole"
        elif mode == 2:
            q.blowupPrefs.defaultSingType = "sing"
        ChebfunPref.setDefaults(q)
    return old
