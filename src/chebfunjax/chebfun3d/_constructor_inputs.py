"""MATLAB-order positional Chebfun3 constructor inputs.

Provenance
----------
MATLAB source: @chebfun3/constructor.m, parseInputs
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref


def parse_source_args(args, domain, pref):
    """Parse selectors and retain the domain at each fixed-length sample."""
    preferences = [arg for arg in args if isinstance(arg, ChebfunPref)]
    if len(preferences) > 1:
        raise ValueError("Only one positional preference object is supported")
    if preferences:
        pref = preferences[0]
    selected_pref = ChebfunPref() if pref is None else ChebfunPref(pref)
    args = [arg for arg in args if not isinstance(arg, ChebfunPref)]
    result = {"pref": selected_pref}
    samples = []
    active_domain = tuple(domain)
    for index, arg in enumerate(args):
        if isinstance(arg, str):
            key = arg.lower()
            if key in ("trig", "periodic"):
                result["trig"] = True
            elif key in ("eps", "rank", "fiberdim"):
                target = "fiberDim" if key == "fiberdim" else key
                result[target] = args[index + 1]
            elif key in ("classic", "chebfun3f"):
                selected_pref.cheb3Prefs.constructor = key
                result["chebfun3f"] = key == "chebfun3f"
            elif key in ("vectorize", "vectorise"):
                result["vectorize"] = True
            elif key in ("coeffs", "equi"):
                result[key] = True
            continue
        array = jnp.asarray(arg)
        if array.dtype.kind not in "iufc":
            continue
        if array.size == 6:
            active_domain = tuple(float(v) for v in array.reshape(-1, order="F"))
            result["domain"] = active_domain
        elif array.size == 3:
            if index and isinstance(args[index - 1], str) and args[index - 1].lower() == "rank":
                continue
            values = array.reshape(-1, order="F").tolist()
            lengths = tuple(int(v) for v in values)
            if any(n != v for n, v in zip(lengths, values)):
                raise ValueError("Fixed lengths must be integers")
            samples.append((lengths, active_domain))
        # Native source ignores numeric arguments of other sizes, including
        # the two length-2 arrays in test_construnctorsyntax's split-domain call.
    return result, samples
