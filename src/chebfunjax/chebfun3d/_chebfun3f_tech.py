"""Selected-technology primitives for the native three-phase constructor.

Provenance
----------
MATLAB source: @chebfun3/chebfun3f.m
Chebfun commit: 7574c77
"""

import jax
import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.utils._binary64 import _divide_binary64_by_positive_integer


def _resolved_happiness_pref(pref, tech, happiness_check):
    """Return the selected technology prefs with the requested checker."""
    if isinstance(pref, dict):
        resolved = dict(pref)
    else:
        owner = ChebfunPref() if pref is None else ChebfunPref(pref)
        if tech.__name__ == "Trigtech":
            owner.tech = "trigtech"
        elif tech.__name__ == "Chebtech1":
            owner.tech = "chebtech1"
        else:
            owner.tech = "chebtech2"
        resolved = dict(owner.techPrefs)
    resolved["happinessCheck"] = happiness_check
    return resolved


def envelope_happiness(
    values, tech, relative_tol, happiness_check="standard", pref=None, *,
    initial_vscale=None,
):
    """Check the literal source coefficient envelope, including its vscale.

    Provenance
    ----------
    MATLAB source: @chebfun3/chebfun3f.m, happinessCheck3;
        @chebtech/happinessCheck.m; @trigtech/happinessCheck.m
    Chebfun commit: 7574c77
    """
    values = jnp.asarray(values)
    coefficients = tech.vals2coeffs(values)
    envelope = jnp.sum(jnp.abs(coefficients), axis=1)
    scale = jnp.max(jnp.abs(tech.coeffs2vals(envelope)))
    if tech.__name__ == "Trigtech":
        # The source trig dispatcher owns classic/plateau and native strict/
        # loose errors; Trigtech.happiness_check itself is standard-only.
        from chebfunjax.tech._trig_constructor import happiness, resolve_pref
        if callable(happiness_check):
            import equinox as eqx

            # Native trigtech retains sampled values and isReal after the
            # source overwrites its coefficient matrix with a one-column
            # envelope. tree_at intentionally bypasses __post_init__, matching
            # that valid-in-MATLAB but width-mismatched temporary object.
            if initial_vscale is None:
                initial_scale = (jnp.max(jnp.abs(values[:, 0]))
                                 if values.ndim > 1 and values.shape[1]
                                 else jnp.max(jnp.abs(values)))
            else:
                initial_scale = jnp.asarray(initial_vscale)
            checker = tech.from_values(
                values, data={"vscale": initial_scale, "hscale": 1.0}
            )
            envelope_column = envelope[:, None]
            checker = eqx.tree_at(
                lambda item: item.coeffs, checker, envelope_column
            )
            checker_pref = ChebfunPref() if pref is None else ChebfunPref(pref)
            checker_pref.chebfuneps = relative_tol
            result = happiness_check(
                checker, envelope_column,
                {"vscale": scale, "hscale": 1.0}, checker_pref,
            )
            return bool(result[0])
        checker_pref = _resolved_happiness_pref(pref, tech, happiness_check)
        checker_pref = resolve_pref(checker_pref)
        checker_pref["chebfuneps"] = relative_tol
        # happinessCheck3 passes empty DATA after mutating the coefficients, so
        # trigtech.happinessCheck derives the effective scale from the envelope.
        checker = tech.from_coeffs(envelope)
        happy, _ = happiness(
            checker, op=None, values=envelope,
            data={"vscale": scale}, pref=checker_pref,
        )
        return bool(happy)

    if callable(happiness_check):
        # Pinned MATLAB @chebtech/happinessCheck.m:83 invokes
        # `pref.happinessCheck(...)` while requesting two outputs. Native
        # ChebfunPref.subsref has one output, so this public polynomial
        # callback path fails before the callback itself is invoked. The
        # trigtech implementation extracts the handle first and does support
        # callbacks; preserve the native polynomial failure here.
        raise TypeError(
            "native chebtech custom happiness callback fails with too many "
            "outputs through ChebfunPref.subsref"
        )

    name = str(happiness_check).lower()
    if name == "loose":
        raise ValueError(
            "CHEBFUN:CHEBTECH:looseCheck:notDone: This file is not implemented yet."
        )
    if name not in {"standard", "classic", "strict", "plateau"}:
        raise ValueError(f"Unsupported polynomial happiness check {happiness_check!r}")
    return bool(tech.happiness_check(
        envelope, envelope, tol=relative_tol, vscale=scale, check=name
    )[0])


_grid_ratios = jax.jit(_divide_binary64_by_positive_integer, static_argnums=1)


def periodic_points(n, a, b):
    """Return the source half-open physical sampling grid.

    Provenance
    ----------
    MATLAB source: @chebfun3/chebfun3f.m, trigtech chebX/Y/Z branch
    Chebfun commit: 7574c77
    """
    indices = jnp.arange(n, dtype=jnp.float64)
    ratio = _grid_ratios(indices, n)
    return a + ratio * (b - a)


def periodic_factors(values, relative_tol, session_tol):
    """Build numeric Fourier factors and apply both source simplifications.

    Provenance
    ----------
    MATLAB source: @chebfun3/chebfun3f.m, phase3;
        @chebfun/simplify.m; @trigtech/simplify.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.tech.trigtech import Trigtech

    combined = Trigtech.from_values(jnp.asarray(values)).simplify(session_tol)
    scales = jnp.max(jnp.abs(combined.values), axis=0)
    tolerances = relative_tol * jnp.max(scales) / scales
    combined = combined.simplify(tolerances)
    return [Trigtech.from_coeffs(combined.coeffs[:, k],
                                is_real=combined.real_columns[k])
            for k in range(values.shape[1])]
