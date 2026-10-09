"""Ordinary public numeric construction, Chebfun source7574c77.

Sources: @chebfun/constructor.m, @smoothfun/smoothfun.m,
@chebtech/populate.m, @chebtech1/chebtech1.m, @chebtech2/chebtech2.m,
@unbndfun/unbndfun.m. Host shape dispatch owns construction; all sample
arithmetic is JAX. Low-level Tech.from_values retains its pure transform API.
"""
import math

import jax.numpy as jnp

from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2, _extrapolate_values
from chebfunjax.tech.trigtech import Trigtech
from chebfunjax.utils.quadrature import chebpts


def source_numeric_chebfun(values, domain, *, tech, n, pref,
                           explicit_trig, zero_overrides=None):
    """Source numeric populate on each interval, including native failures."""
    from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
    from chebfunjax.fun.unbndfun import Unbndfun

    points = tuple(float(x) for x in domain)
    if values.size == 0 or not points or n == 0:
        return Chebfun.empty()
    if values.ndim > 2:
        raise ValueError('Numeric values must be scalar, vector or matrix.')
    dom = Domain(points)
    key = (tech.__name__ if isinstance(tech, type) else str(tech)).lower().lstrip('@')
    classes = {'chebtech': Chebtech2, 'chebtech1': Chebtech1,
               'chebtech2': Chebtech2, 'trigtech': Trigtech}
    if key not in classes:
        raise ValueError(f'Unknown numeric construction Tech: {tech!r}')
    cls = classes[key]
    if cls is Trigtech and any(
            (zero_overrides or {}).get(name) is not None
            for name in ("sample_test", "refinement_function")):
        raise ValueError("sample_test/refinement_function overrides are not "
                         "yet supported for numeric Trigtech construction")
    if explicit_trig and len(points) != 2:
        raise ValueError('CHEBFUN:parseInputs:periodic: periodic construction '
                         'does not support domains with breakpoints.')
    length = n if n is not None else pref.techPrefs.fixedLength
    if length is not None and not math.isnan(float(length)):
        numeric_length = float(length)
        if not math.isfinite(numeric_length) or not numeric_length.is_integer() or numeric_length < 1:
            raise ValueError('Numeric fixedLength must be a positive integer.')
        length = int(numeric_length)
    else:
        length = None
    values = jnp.atleast_1d(values)
    values = values.astype(jnp.complex128 if jnp.iscomplexobj(values) else jnp.float64)
    pieces = []
    for a, b in zip(points[:-1], points[1:]):
        interval = Domain((a, b))
        if math.isinf(a) or math.isinf(b):
            if bool(jnp.any(values != 0)):
                raise ValueError('CHEBFUN:UNBNDFUN:unbndfun:inputValues: '
                                 'UNBNDFUN does not support non-zero construction from values.')
            # Native zero numeric input becomes a zero-valued operator.
            # Preserve its column count and selected Tech at that boundary.
            def zero(x):
                shape = (x.shape[0],) + values.shape[1:]
                return jnp.zeros(shape, dtype=jnp.float64)
            options = {}
            if cls is not Trigtech:
                # smoothfun forwards resolved techPrefs to the selected Tech.
                # Explicit keyword values win, including explicit False.
                options = {
                    "tol": pref.techPrefs.chebfuneps,
                    "check": pref.techPrefs.happinessCheck,
                    "max_length": pref.techPrefs.maxLength,
                    "min_samples": pref.techPrefs.minSamples,
                    "sample_test": pref.techPrefs.sampleTest,
                    "refinement_function": pref.techPrefs.refinementFunction,
                    "turbo": pref.techPrefs.get("useTurbo", False),
                }
                if cls is Chebtech2:
                    options["extrapolate"] = pref.techPrefs.extrapolate
                for key, value in (zero_overrides or {}).items():
                    if value is not None and key in options:
                        options[key] = value
            # Trigtech's existing n/maxpow2 API cannot express these prefs.
            # Retain its qualified numeric/fixedLength behavior separately.
            constructed = cls.from_function(zero, n=length, **options)
            pieces.append(Unbndfun.from_chebtech(constructed, interval))
            continue
        sampled = values
        if cls is not Trigtech and not bool(jnp.all(jnp.isnan(sampled))):
            if not bool(jnp.all(jnp.isfinite(sampled))):
                kind = 1 if cls is Chebtech1 else 2
                sampled = _extrapolate_values(
                    sampled, chebpts(sampled.shape[0], kind=kind),
                    cls.barywts(sampled.shape[0]))[0]
        constructed = cls.from_values(sampled)
        # Native numeric populate first, then fixedLength prolongation.
        if length is not None:
            constructed = constructed.prolong(length)
        pieces.append(_Piece(tech=constructed, interval=(a, b)))
    return Chebfun(funs=pieces, domain=dom)
