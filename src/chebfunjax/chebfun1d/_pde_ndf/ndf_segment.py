"""Multi-step NDF segment with native controller semantics.

Source: MATLAB R2025b ode15s main/error/acceptance/controller loops.
Constant full mass, positive time, componentwise control. Newton-too-slow
recovery, Jacobian refresh, dense output, and spatial callbacks are supported.

Provenance
----------
MATLAB R2025b ode15s.m rejection, recovery, accepted-step and output loops.
Chebfun native commit: 7574c77680d7e82b79626300bf255498271a72df.
Native ODE runtime/source release: MATLAB R2025b.
"""

import warnings

import jax.numpy as jnp
import jax.scipy.linalg as jl

from .dae_init import dense_numjac
from .ndf import positive_max, sixteen_eps
from .ndf_attempt import attempt
from .ndf_controller import accepted_controller, rescale
from .ndf_stages import accepted_dif, dense_output


def segment(initial, fun, max_attempts, output_times, callback, *, record_trace=True):
    s = dict(initial)
    s.update(
        k=1,
        nconhk=0,
        abshlast=s["absh"],
        klast=1,
        havrate=False,
        rate=0.0,
        done=False,
        Jcurrent=True,
        at_hmin=False,
    )
    rows = []
    outer = True
    nofailed = True
    next_output = 0

    def tolerance_failure():
        warnings.warn(
            "MATLAB:ode15s:IntegrationTolNotMet: Failure at t=%e. "
            "Unable to meet integration tolerances without reducing the step "
            "size below the smallest value allowed (%e) at time t."
            % (float(s["t"]), float(s["hmin"])),
            RuntimeWarning,
            stacklevel=2,
        )
        s["terminal_reason"] = "IntegrationTolNotMet"
        s["callback_stopped"] = False
        return s, rows

    def stretch(previous):
        update = rescale(
            s["dif"],
            s["absh"],
            previous,
            s["k"],
            s["mass"],
            s["jac"],
            s["invGa"][s["k"] - 1],
            dae=s.get("dae", True),
        )
        s.update(update)
        s["Factors"], s["piv"] = jl.lu_factor(s["Miter"])
        s["nconhk"] = 0
        s["havrate"] = False

    number = 0
    while max_attempts is None or number < max_attempts:
        number += 1
        s["attempt_count"] = number
        if outer:
            s["hmin"] = positive_max(sixteen_eps(s["t"]), s.get("userhmin", 0.0))
            s["hmax"] = positive_max(sixteen_eps(s["t"]), s.get("userhmax", s["hmax"]))
            s["absh"] = jnp.minimum(s["hmax"], positive_max(s["hmin"], s["absh"]))
            if bool(s["absh"] == s["hmin"]):
                if s["at_hmin"]:
                    s["absh"] = s["abshlast"]
                s["at_hmin"] = True
            else:
                s["at_hmin"] = False
            s["h"] = s["absh"]
            if bool(1.1 * s["absh"] >= jnp.abs(s["tfinal"] - s["t"])):
                s["h"] = s["tfinal"] - s["t"]
                s["absh"] = jnp.abs(s["h"])
                s["done"] = True
            if bool(s["absh"] != s["abshlast"]) or s["k"] != s["klast"]:
                stretch(s["abshlast"])
            nofailed = True
        used_k = s["k"]
        input_havrate = s["havrate"]
        recoveries = []
        while True:
            out = attempt(s, fun)
            s["rate"] = out["rate"]
            s["havrate"] = out["havrate"]
            if not out["tooslow"]:
                break
            if not s["Jcurrent"]:

                def at_current(y):
                    return fun(s["t"], y)

                value = at_current(s["y"])
                s["jac"], s["jac_fac"], _ = dense_numjac(
                    at_current, s["y"], value, s["jac_threshold"], s.get("jac_fac")
                )
                s["Jcurrent"] = True
                recoveries.append("refresh_jacobian")
                matrix = s["mass"] - s["hinvGak"] * s["jac"]
                s["RowScale"] = (
                    1 / jnp.max(jnp.abs(matrix), axis=1)
                    if s.get("dae", True)
                    else jnp.ones_like(s["y"])
                )
                s["Miter"] = s["RowScale"][:, None] * matrix
                s["Factors"], s["piv"] = jl.lu_factor(s["Miter"])
                s["havrate"] = False
            elif bool(s["absh"] <= s["hmin"]):
                return tolerance_failure()
            else:
                previous = s["absh"]
                s["abshlast"] = previous
                s["absh"] = positive_max(0.3 * previous, s["hmin"])
                s["h"] = s["absh"]
                s["done"] = False
                stretch(previous)
                recoveries.append("reduce_step")
        row = dict(
            newton_recoveries=recoveries,
            attempt=number,
            input_havrate=input_havrate,
            k=used_k,
            h=out["h"],
            tnew=out["tnew"],
            ynew=out["ynew"],
            error=out["error"],
            rejected=out["rejected"],
            iterations=len(out["log"]),
            reason=out["reason"],
        )
        if record_trace:
            rows.append(row)
        if out["rejected"]:
            previous = s["absh"]
            s["abshlast"] = previous
            if bool(previous <= s["hmin"]):
                return tolerance_failure()
            if nofailed:
                nofailed = False
                hopt = previous * jnp.maximum(
                    0.1, 0.833 * (s["rtol"] / out["error"]) ** (1 / (used_k + 1))
                )
                if used_k > 1:
                    lower = (
                        jnp.max(jnp.abs((s["dif"][:, used_k - 1] + out["difkp1"]) * out["invwt"]))
                        * s["erconst"][used_k - 2]
                    )
                    hkm1 = previous * jnp.maximum(0.1, 0.769 * (s["rtol"] / lower) ** (1 / used_k))
                    if bool(hkm1 > hopt):
                        hopt = jnp.minimum(previous, hkm1)
                        s["k"] -= 1
                s["absh"] = positive_max(s["hmin"], hopt)
            else:
                s["absh"] = positive_max(s["hmin"], 0.5 * previous)
            s["h"] = s["absh"]
            if bool(s["absh"] < previous):
                s["done"] = False
            stretch(previous)
            outer = False
        else:
            s["dif"] = accepted_dif(s["dif"], out["difkp1"], used_k)
            row["accepted_dif"] = s["dif"]
            row["invwt"] = out["invwt"]
            row["outputs"] = []
            while next_output < len(output_times) and bool(
                output_times[next_output] <= out["tnew"]
            ):
                target = output_times[next_output]
                value = dense_output(
                    jnp.asarray(target), out["tnew"], out["ynew"], out["h"], s["dif"], used_k
                )[:, 0]
                row["outputs"].append((target, value))
                next_output += 1
                if callback(target, value):
                    s["t"] = out["tnew"]
                    s["y"] = out["ynew"]
                    s["callback_stopped"] = True
                    return s, rows
            if s["done"]:
                s["t"] = out["tnew"]
                s["y"] = out["ynew"]
                s["callback_stopped"] = False
                return s, rows
            s["klast"] = used_k
            s["abshlast"] = s["absh"]
            lower = higher = None
            if s["nconhk"] + 1 >= used_k + 2:
                if used_k > 1:
                    lower = (
                        jnp.max(jnp.abs(s["dif"][:, used_k - 1] * out["invwt"]))
                        * s["erconst"][used_k - 2]
                    )
                if used_k < 5:
                    higher = (
                        jnp.max(jnp.abs(s["dif"][:, used_k + 1] * out["invwt"]))
                        * s["erconst"][used_k]
                    )
            s.update(
                accepted_controller(
                    s["absh"], used_k, s["nconhk"], out["error"], s["rtol"], lower, higher
                )
            )
            row["next_k"] = s["k"]
            row["next_absh"] = s["absh"]
            row["nconhk"] = s["nconhk"]
            s["t"] = out["tnew"]
            s["y"] = out["ynew"]
            s["Jcurrent"] = False
            outer = True
            if s["done"]:
                break
    return s, rows
