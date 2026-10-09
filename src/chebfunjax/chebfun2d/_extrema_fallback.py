"""Source higher-rank seed and optimizer transaction, not an active-set solver.

Caller must supply the source fixed4000-converted, simplified and pivot-scaled
row/column arrays. Public dispatch awaits constructor and wrapper qualification.
Absent active-set capability goes directly to the native fallback branch; no
exception is fabricated and the legacy SciPy optimizer is not substituted.

Provenance: @separableApprox/minandmax2.m, @mapping/mapping.m and fevalm/cdr.m,
Chebfun7574c77680d7e82b79626300bf255498271a72df. Copyright University of Oxford
and the Chebfun Developers. Fallback kernel targets licensed MATLAB R2017a.
"""

import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from chebfunjax.chebfun2d._cdr_source import _mesh_values
from chebfunjax.utils._fminsearch import fminsearch
from chebfunjax.utils.quadrature import chebpts


@dataclass(frozen=True)
class ActiveOptions:
    """Exact explicitly supplied native active-set preferences."""

    display: str = "none"
    tol_fun: float = 2.0**-52
    tol_x: float = 2.0**-52
    algorithm: str = "active-set"


def _forward(t, a, b):
    return b * (t + 1) / 2 + a * (1 - t) / 2


def _inverse(x, a, b):
    return (x - a) / (b - a) - (b - x) / (b - a)


def _mapped(z, domain):
    a, b, c, d = domain
    return jnp.stack((_forward(jnp.sin(z[0]), a, b),
                      _forward(jnp.sin(z[1]), c, d)))


def original_objective(approx):
    """Compile original CDR scalar action once, not reconstructed seed values."""
    @jax.jit
    def objective(z):
        return _mesh_values(approx, z[0:1], z[1:2]).reshape(())

    return objective


def source_seed(rows, cols, objective):
    """Native sample lengths and first column-major min/max, no conjugation."""
    xpts = _forward(chebpts(len(rows)), rows.domain.a, rows.domain.b)
    ypts = _forward(chebpts(len(cols)), cols.domain.a, cols.domain.b)
    cvals = jnp.asarray(cols(ypts)).reshape((len(cols), -1))
    rvals = jnp.asarray(rows(xpts)).reshape((len(rows), -1))
    values = cvals @ rvals.T
    flat = values.reshape(-1, order="F")
    indices = jnp.stack((jnp.argmin(flat), jnp.argmax(flat)))
    locations = jnp.stack((xpts[indices // len(cols)],
                           ypts[indices % len(cols)]), axis=1)
    return jnp.stack((objective(locations[0]), objective(locations[1]))), locations


def refine_seed(objective, values, locations, domain, *, active_solver=None,
                fallback_solver=fminsearch):
    """Literal partial assignments; optimizer status never triggers fallback.

    Injected active solver contract: (objective, x0, lower, upper, options) ->
    an object with x/fun fields. Status fields are intentionally ignored. The
    optional second argument of native first objective has no used value; the
    Python callback adapter accepts either invocation, without inventing a
    missing-argument failure. Only Exception is caught: resource BaseException
    always escapes. Scoped warning restoration is a Python host adaptation.
    """
    y, x = jnp.asarray(values), jnp.asarray(locations)
    if active_solver is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a, b, c, d = domain
                lower, upper = jnp.asarray((a, c)), jnp.asarray((b, d))
                mn = active_solver(lambda z, *unused: objective(z), x[0],
                                   lower, upper, ActiveOptions())
                y = y.at[0].set(mn.fun)
                mx = active_solver(lambda z: -objective(z), x[1],
                                   lower, upper, ActiveOptions())
                y = y.at[1].set(mx.fun)
                y = y.at[1].set(-y[1])
                x = x.at[0].set(mn.x)
                x = x.at[1].set(mx.x)
            return y, x
        except Exception:
            pass
    try:
        a, b, c, d = domain
        z = jnp.stack((jnp.arcsin(_inverse(x[:, 0], a, b)),
                       jnp.arcsin(_inverse(x[:, 1], c, d))), axis=1)
        mapped_objective = jax.jit(lambda zz: objective(_mapped(zz, domain)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mn = fallback_solver(mapped_objective, z[0])
            y = y.at[0].set(mn.fun)
            mx = fallback_solver(lambda zz: -mapped_objective(zz), z[1])
            y = y.at[1].set(mx.fun)
            y = y.at[1].set(-y[1])
            x = x.at[:, 0].set(_forward(jnp.sin(jnp.stack((mn.x[0], mx.x[0]))), a, b))
            x = x.at[:, 1].set(_forward(jnp.sin(jnp.stack((mn.x[1], mx.x[1]))), c, d))
    except Exception:
        pass
    return y, x


def source_higher_extrema(approx, rows, cols, *, active_solver=None,
                          fallback_solver=fminsearch):
    """Consume already-converted/scaled arrays; not public preference routing."""
    if approx.rank <= 1:
        raise ValueError("Higher-rank transaction requires rank greater than one")
    if approx.rank > 4000:
        raise ValueError("Rank is too large")
    objective = original_objective(approx)
    y, x = source_seed(rows, cols, objective)
    return refine_seed(objective, y, x, approx.domain, active_solver=active_solver,
                       fallback_solver=fallback_solver)
