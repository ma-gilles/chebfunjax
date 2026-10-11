"""NDF Newton iteration and error stage used by the public solver.
MATLAB R2025b ode15s.m lines 513-608; constant full mass, real64,
componentwise error, no nonnegative state or outer JIT qualification.

Provenance
----------
MATLAB R2025b ode15s.m Newton iteration and local-error estimation.
Chebfun native commit: 7574c77680d7e82b79626300bf255498271a72df.
Native ODE runtime/source release: MATLAB R2025b.
"""

import jax.numpy as jnp
import jax.scipy.linalg as jl

from .ndf import EPS
from .ndf_controller import ordered_product


def attempt(s, fun):
    """One source generic-order Newton/error attempt, constant full mass."""
    y = s["y"]
    h = s["h"]
    rtol = s["rtol"]
    dif = s["dif"]
    mass = s["mass"]
    k = int(s["k"])
    psi = ordered_product(dif[:, :k], (s["G"][:k] * s["invGa"][k - 1])[:, None])[:, 0]
    tnew = s["t"] + h
    if s.get("done", False):
        tnew = s["tfinal"]
    h = tnew - s["t"]
    pred = y + ordered_product(dif[:, :k], jnp.ones((k, 1), dtype=y.dtype))[:, 0]
    ynew = pred
    difkp1 = jnp.zeros_like(y)
    invwt = 1 / jnp.maximum(jnp.maximum(jnp.abs(y), jnp.abs(ynew)), s["threshold"])
    minnrm = 100 * EPS * jnp.max(jnp.abs(ynew * invwt))
    rate = s.get("rate", 0.0)
    havrate = bool(s.get("havrate", False))
    log = []
    reason = "maxit"
    tooslow = False
    factor = (s["Factors"], s["piv"])
    oldnrm = 0.0  # Read only after the first iteration assigns its norm.
    for iteration in range(1, 5):
        value = fun(tnew, ynew)
        first = s["hinvGak"] * value
        second = mass @ (psi + difkp1)
        rhs = first - second
        rhs = s["RowScale"] * rhs
        delta = jl.lu_solve(factor, rhs)
        newnrm = jnp.max(jnp.abs(delta * invwt))
        difkp1 = difkp1 + delta
        ynew = pred + difkp1
        log.append(
            dict(
                iteration=iteration,
                input_value=value,
                rhs=rhs,
                delta=delta,
                ynew=ynew,
                difkp1=difkp1,
                newnrm=newnrm,
                minnrm=minnrm,
            )
        )
        if bool(newnrm <= minnrm):
            reason = "minnrm"
            break
        elif iteration == 1:
            if havrate:
                errit = newnrm * rate / (1 - rate)
                if bool(errit <= 0.05 * rtol):
                    reason = "old_rate"
                    break
            else:
                rate = 0.0
        elif bool(newnrm > 0.9 * oldnrm):
            tooslow = True
            reason = "slow"
            break
        else:
            rate = jnp.maximum(0.9 * rate, newnrm / oldnrm)
            havrate = True
            errit = newnrm * rate / (1 - rate)
            if bool(errit <= 0.5 * rtol):
                reason = "rate"
                break
            elif iteration == 4:
                tooslow = True
                reason = "maxit"
                break
            elif bool(0.5 * rtol < errit * rate ** (4 - iteration)):
                tooslow = True
                reason = "rate_prediction"
                break
        oldnrm = newnrm
    error = jnp.max(jnp.abs(difkp1 * invwt)) * s["erconst"][k - 1]
    return dict(
        invwt=invwt,
        tnew=tnew,
        h=h,
        pred=pred,
        psi=psi,
        log=log,
        ynew=ynew,
        difkp1=difkp1,
        error=error,
        rejected=bool(error > rtol),
        tooslow=tooslow,
        reason=reason,
        rate=rate,
        havrate=havrate,
    )
