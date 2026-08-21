"""MATLAB-shaped preference objects: chebfunpref / cheboppref.

The runtime preference system chebfunjax actually consults lives in
:mod:`chebfunjax.pref`; this module ports the MATLAB *object* semantics
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


def _factory_tech_prefs() -> DotDict:
    return DotDict.wrap({
        "chebfuneps": _EPS,
        "maxLength": 65537,
        "minSamples": 17,
        "fixedLength": None,
        "extrapolate": False,
        "sampleTest": True,
        "refinementFunction": "nested",
        "happinessCheck": "standard",
    })


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


class ChebfunPref:
    """MATLAB ``chebfunpref``: top-level structure fields plus a
    ``techPrefs`` substructure; unknown names route into techPrefs on
    both read and write.

    Provenance
    ----------
    MATLAB source : @chebfunpref/chebfunpref.m
    Chebfun commit: 7574c77
    """

    _defaults: "ChebfunPref | None" = None

    def __init__(self, src=None, **kwargs):
        base = type(self).__dict__.get("_defaults")
        if base is not None and src is None and not kwargs:
            self.__dict__["_top"] = copy.deepcopy(base._top)
            self.__dict__["techPrefs"] = copy.deepcopy(base.techPrefs)
            return
        self.__dict__["_top"] = _factory_top()
        self.__dict__["techPrefs"] = _factory_tech_prefs()
        if isinstance(src, ChebfunPref):
            self.__dict__["_top"] = copy.deepcopy(src._top)
            self.__dict__["techPrefs"] = copy.deepcopy(src.techPrefs)
        elif isinstance(src, dict):
            self._absorb(src)
        if kwargs:
            self._absorb(kwargs)

    def _absorb(self, d: dict):
        for k, v in d.items():
            if k == "techPrefs" and isinstance(v, dict):
                for tk, tv in v.items():
                    self.techPrefs[tk] = (DotDict.wrap(tv)
                                          if isinstance(tv, dict)
                                          else tv)
            elif k in self._top:
                self._top[k] = (DotDict.wrap(v)
                                if isinstance(v, dict)
                                and isinstance(self._top[k], dict)
                                else v)
            else:
                self.techPrefs[k] = (DotDict.wrap(v)
                                     if isinstance(v, dict) else v)

    def __getattr__(self, name):
        top = self.__dict__["_top"]
        if name in top:
            return top[name]
        tp = self.__dict__["techPrefs"]
        if name in tp:
            return tp[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name == "techPrefs":
            self.__dict__["techPrefs"] = (
                DotDict.wrap(value) if isinstance(value, dict)
                else value)
        elif name in self.__dict__["_top"]:
            self.__dict__["_top"][name] = value
        else:
            self.__dict__["techPrefs"][name] = value

    def __eq__(self, other):
        return (isinstance(other, ChebfunPref)
                and self._top == other._top
                and self.techPrefs == other.techPrefs)

    __hash__ = None

    # -- statics -------------------------------------------------------

    @staticmethod
    def mergeTechPrefs(p, q) -> DotDict:
        """Merge two techPrefs structures (later wins); ChebfunPref
        inputs contribute their ``techPrefs``.

        Provenance
        ----------
        MATLAB source : @chebfunpref/mergeTechPrefs.m
        Chebfun commit: 7574c77
        """
        def tp(x):
            return (copy.deepcopy(x.techPrefs)
                    if isinstance(x, ChebfunPref)
                    else DotDict.wrap(dict(x)))
        out = tp(p)
        out.update(tp(q))
        return out

    @classmethod
    def getFactoryDefaults(cls) -> "ChebfunPref":
        """The factory-default preference object.

        Provenance
        ----------
        MATLAB source : @chebfunpref/getFactoryDefaults.m
        Chebfun commit: 7574c77
        """
        saved = cls._defaults
        cls._defaults = None
        try:
            return cls()
        finally:
            cls._defaults = saved

    @classmethod
    def setDefaults(cls, *args, **kwargs):
        """Set (or factory-reset) the session default preferences:
        ``setDefaults('factory')``, ``setDefaults(prefOrDict)``, or
        name/value pairs (a value of ``'factory'`` resets that name).

        Provenance
        ----------
        MATLAB source : @chebfunpref/setDefaults.m
        Chebfun commit: 7574c77
        """
        if len(args) == 1 and args[0] == "factory" and not kwargs:
            cls._defaults = None
            return
        if len(args) == 1 and isinstance(args[0],
                                         (ChebfunPref, dict)):
            cls._defaults = cls(args[0])
            return
        pairs = dict(zip(args[0::2], args[1::2]))
        pairs.update(kwargs)
        base = cls()
        factory = cls.getFactoryDefaults()
        for k, v in pairs.items():
            if isinstance(v, str) and v == "factory":
                if k in factory._top:
                    base._top[k] = copy.deepcopy(factory._top[k])
                else:
                    base.techPrefs[k] = factory.techPrefs.get(k)
            else:
                setattr(base, k, v)
        cls._defaults = base


class ChebopPref(ChebfunPref):
    """MATLAB ``cheboppref``: the chebop preference structure.

    Provenance
    ----------
    MATLAB source : @cheboppref/cheboppref.m
    Chebfun commit: 7574c77
    """

    _defaults: "ChebopPref | None" = None

    def __init__(self, src=None, **kwargs):
        super().__init__(src, **kwargs)
        top = self.__dict__["_top"]
        for k, v in (("discretization", "chebcolloc2"),
                     ("bvpTol", 1e-10), ("ivpAbsTol", 1e5 * _EPS),
                     ("ivpRelTol", 100 * _EPS), ("damping", True),
                     ("maxIter", 25), ("plotting", "off"),
                     ("display", "off"),
                     ("ivpSolver", "ode113")):
            top.setdefault(k, v)
        if isinstance(src, dict):
            for k in src:
                if k in top and k not in _factory_top():
                    top[k] = src[k]
