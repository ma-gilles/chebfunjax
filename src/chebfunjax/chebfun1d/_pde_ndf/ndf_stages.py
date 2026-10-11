"""NDF difference updates and dense output used by the public solver.

Source algorithms: MATLAB R2025b ode15s.m (MathWorks copyright 1984-2024)
and private/ntrp15s.m (copyright 1984-2022). Real constant full mass, no
nonnegative components, componentwise norm. Host statement ordering is explicit;
outer JIT is outside qualification.

Provenance
----------
MATLAB R2025b ode15s.m difference updates and private/ntrp15s.m.
Chebfun native commit: 7574c77680d7e82b79626300bf255498271a72df.
Native ODE runtime/source release: MATLAB R2025b.
"""

import jax.numpy as jnp

from .ndf import positive_max


def cumulative_product_rows(x):
    rows = [x[0]]
    for i in range(1, x.shape[0]):
        rows.append(rows[-1] * x[i])
    return jnp.stack(rows)


def rejected_k1(absh, err, rtol, hmin, dif, mass, jac, invga, *, nofailed):
    """Source first/subsequent error rejection at k=1, positive direction."""
    if nofailed:
        ratio = rtol / err
        hopt = absh * jnp.maximum(0.1, 0.833 * ratio**0.5)
        newh = positive_max(hmin, hopt)
    else:
        newh = positive_max(hmin, 0.5 * absh)
    ki = jnp.arange(1, 6, dtype=jnp.float64)[:, None]
    kj = jnp.arange(1, 6, dtype=jnp.float64)[None, :]
    u = jnp.array(
        [
            [-1, -2, -3, -4, -5],
            [0, 1, 3, 6, 10],
            [0, 0, -1, -4, -10],
            [0, 0, 0, 1, 5],
            [0, 0, 0, 0, -1],
        ],
        dtype=jnp.float64,
    )
    ru = cumulative_product_rows((ki - 1 - kj * (newh / absh)) / ki) @ u
    updated = dif.at[:, :1].set(dif[:, :1] @ ru[:1, :1])
    hinv = newh * invga
    matrix = mass - hinv * jac
    scale = 1 / jnp.max(jnp.abs(matrix), axis=1)
    matrix = scale[:, None] * matrix
    return dict(
        h=newh, absh=newh, dif=updated, difRU=ru, hinvGak=hinv, RowScale=scale, Miter=matrix
    )


def accepted_dif(dif, difkp1, k):
    out = dif.at[:, k + 1].set(difkp1 - dif[:, k])
    out = out.at[:, k].set(difkp1)
    for j in range(k - 1, -1, -1):
        out = out.at[:, j].set(out[:, j] + out[:, j + 1])
    return out


def dense_output(tinterp, tnew, ynew, h, dif, k):
    """Value-only ntrp15s route; no nonnegative components or derivative."""
    s = (jnp.atleast_1d(tinterp) - tnew) / h
    if k == 1:
        weights = s[None, :]
    else:
        ki = jnp.arange(1, k + 1, dtype=jnp.float64)[:, None]
        weights = cumulative_product_rows((s[None, :] + ki - 1) / ki)
    return ynew[:, None] + dif[:, :k] @ weights
