"""Continuous Lebesgue functions for polynomial and trigonometric nodes.

Provenance
----------
MATLAB source: lebesgue.m, baryWeights.m, trigBaryWeights.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
Copyright The University of Oxford and The Chebfun Developers.
"""
import math

import jax
import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.utils.interpolation import bary_weights, trig_bary_weights


def _numeric(value):
    try:
        return jnp.asarray(value).dtype.kind in "iufc"
    except (TypeError, ValueError):
        return False


def _domain_pair(values):
    arrays = []
    for value in values:
        a = jnp.asarray(value)
        if not a.size:
            continue
        arrays.append(a.reshape(1, -1) if a.ndim < 2 else a)
    return jnp.concatenate(arrays, axis=1) if arrays else jnp.empty((0, 0))


def _parse(nodes, args):
    domain, trig = (-1., 1.), False
    is_trig = lambda arg: isinstance(arg, str) and arg.lower() == "trig"  # noqa: E731
    if len(args) == 1:
        if _numeric(args[0]):
            domain = args[0]
        elif is_trig(args[0]):
            trig = True
        else:
            raise ValueError("CHEBFUN:lebesgue:parseInputs:badArg1")
    elif len(args) == 2:
        if _numeric(args[0]) and _numeric(args[1]):
            domain = _domain_pair(args)
        elif _numeric(args[0]) and is_trig(args[1]):
            domain, trig = args[0], True
        else:
            raise ValueError("CHEBFUN:lebesgue:parseInputs:badArg2")
    elif len(args) == 3:
        if _numeric(args[0]) and _numeric(args[1]) and is_trig(args[2]):
            domain = _domain_pair(args[:2])
            trig = True
        else:
            raise ValueError("CHEBFUN:lebesgue:parseInputs:badArg3")
    elif len(args) > 3:
        raise ValueError("CHEBFUN:lebesgue:parseInputs:tooManyArgs")
    d = jnp.asarray(domain)
    if d.shape not in ((2,), (1, 2)):
        raise ValueError("CHEBFUN:lebesgue:parseInputs:badDom")
    a, b = (float(v) for v in d.ravel())
    if not a < b:
        raise ValueError("CHEBFUN:lebesgue:parseInputs:badDom")
    x = jnp.asarray(nodes)
    if bool(jnp.any(x < a-10*math.ulp(a))) or bool(jnp.any(x > b+10*math.ulp(b))):
        raise ValueError("CHEBFUN:lebesgue:parseInputs:pointsOutsideDomain")
    return x, (a, b), trig


def _weights(x, trig):
    if x.ndim > 2 or (x.ndim == 2 and min(x.shape) > 1):
        name = "trigBaryWts" if trig else "baryWeights"
        raise ValueError(f"CHEBFUN:{name}:matrix: Input must be a vector.")
    x = x.ravel(order="F")
    n = len(x)
    if trig and bool(jnp.all(jnp.abs(jnp.diff(x)-2*jnp.pi/n)
                            < jnp.max(jnp.abs(x))*jnp.finfo(jnp.float64).eps)):
        return jnp.where(jnp.arange(n) % 2, -1., 1.)
    if n < 2001:
        return trig_bary_weights(x) if trig else bary_weights(x)
    capacity = 1. if trig or jnp.iscomplexobj(x) else 4/(jnp.max(x)-jnp.min(x))

    def one(index):
        delta = x[index]-x
        v = jnp.sin(.5*delta) if trig else capacity*delta
        v = v.at[index].set(1.)
        return 1/(jnp.prod(jnp.sign(v))*jnp.exp(jnp.sum(jnp.log(jnp.abs(v)))))

    weights = jax.lax.map(one, jnp.arange(n))
    return weights/jnp.max(jnp.abs(weights))


def _values(t, nodes, weights, trig=False):
    t = jnp.asarray(t)
    delta = t[..., None]-nodes
    member = jnp.any(delta == 0, axis=-1)
    denominator = jnp.sin(delta/2) if trig else delta
    # Native skips exact nodes. Safe unselected denominators preserve L(x)=1.
    denominator = jnp.where(delta == 0, 1., denominator)
    ratios = weights/denominator
    value = jnp.sum(jnp.abs(ratios), axis=-1)/jnp.abs(jnp.sum(ratios, axis=-1))
    return jnp.where(member, 1., value)


def lebesgue(nodes, *args, return_constant=False):
    """Return the continuous Lebesgue function and optionally its inf norm.

    Accept ``nodes``, ``nodes, (a,b)``, or ``nodes, a,b``, optionally followed
    by ``"trig"``. ``return_constant=True`` returns ``(L, Lconst)``.
    Construction is eager; evaluation of the resulting Chebfun uses JAX.

    Provenance
    ----------
    MATLAB source: lebesgue.m, baryWeights.m, trigBaryWeights.m.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    """
    from chebfunjax.chebfun1d.chebfun import chebfun

    x, (a, b), trig = _parse(nodes, args)
    preference = ChebfunPref()
    tech = preference.tech
    tech_name = tech.__name__ if isinstance(tech, type) else str(tech)
    polynomial = tech_name.lower().lstrip("@") in ("chebtech", "chebtech1", "chebtech2")
    options = dict(tech=tech, eps=float(preference.techPrefs.chebfuneps))
    if polynomial:
        options.update(n=preference.techPrefs.fixedLength,
                       max_length=preference.techPrefs.maxLength,
                       min_samples=preference.techPrefs.minSamples,
                       refinement_function=preference.techPrefs.refinementFunction,
                       sample_test=preference.techPrefs.sampleTest)
    # Keep original nodes for breakpoint construction, including a periodic end.
    domain = tuple(float(v) for v in jnp.unique(jnp.concatenate(
        (x.ravel(order="F"), jnp.asarray([a, b])))))
    if trig:
        mapped = jnp.pi/(b-a)*(2*x-a-b)
        flat = mapped.ravel(order="F")
        weight_nodes = mapped
        if bool(jnp.max(jnp.abs(jnp.asarray([flat[0], flat[-1]])
                               -jnp.asarray([-jnp.pi, jnp.pi]))) < 2*jnp.pi*jnp.finfo(jnp.float64).eps):
            flat = flat[:-1]
            weight_nodes = flat
        count = max(weight_nodes.shape, default=1)
        if count % 2 == 0:
            raise ValueError("CHEBFUN:lebesgue:trigLebesgue:evenLengthGrid")
        weights = _weights(weight_nodes, True)
        L = chebfun(lambda t: _values(jnp.pi/(b-a)*(2*t-a-b), flat, weights, True),
                    domain=domain, **options)
    else:
        weights = _weights(x, False)
        flat = x.ravel(order="F")
        options["sample_test"] = False
        if polynomial:
            options["n"] = len(flat)
        L = chebfun(lambda t: _values(t, flat, weights), domain=domain, **options)
        if polynomial:
            L = L.simplify(float(preference.techPrefs.chebfuneps))
    return (L, L.norm(jnp.inf)) if return_constant else L
