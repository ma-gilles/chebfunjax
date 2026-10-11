"""Private NDF controller stages from MATLAB R2025b ode15s.m.

MathWorks source copyright1984-2024. Constant full mass, real componentwise
error control, maxorder5; exact saved scalar errors supplied independently.
Used by the public NDF segment with eager source statement ordering.

Provenance
----------
MATLAB R2025b ode15s.m accepted-step order/step-size controller and rescaling.
Chebfun native commit: 7574c77680d7e82b79626300bf255498271a72df.
Native ODE runtime/source release: MATLAB R2025b.
"""

import jax.numpy as jnp

from .ndf_stages import cumulative_product_rows


def accepted_controller(absh, k, nconhk, err, rtol, errkm1=None, errkp1=None):
    nconhk = min(nconhk + 1, 7)
    if nconhk >= k + 2:
        temp = 1.2 * (err / rtol) ** (1 / (k + 1))
        hopt = absh / temp if bool(temp > 0.1) else 10 * absh
        kopt = k
        if k > 1:
            temp = 1.3 * (errkm1 / rtol) ** (1 / k)
            hkm1 = absh / temp if bool(temp > 0.1) else 10 * absh
            if bool(hkm1 > hopt):
                hopt = hkm1
                kopt = k - 1
        if k < 5:
            temp = 1.4 * (errkp1 / rtol) ** (1 / (k + 2))
            hkp1 = absh / temp if bool(temp > 0.1) else 10 * absh
            if bool(hkp1 > hopt):
                hopt = hkp1
                kopt = k + 1
        if bool(hopt > absh):
            absh = hopt
            k = kopt
    return dict(absh=absh, k=k, nconhk=nconhk)


def ordered_product(a, b):
    """Small native NDF products: ascending terms, separate multiply/add."""
    out = jnp.zeros((a.shape[0], b.shape[1]), dtype=jnp.result_type(a, b))
    for i in range(a.shape[1]):
        term = a[:, i, None] * b[None, i, :]
        out = out + term
    return out


def rescale(dif, absh, abshlast, k, mass, jac, invga, dae=True):
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
    numerator = ki - 1 - kj * (absh / abshlast)
    # Materialize equal shapes to preserve elementwise division on eager CPU.
    factors = numerator / jnp.broadcast_to(ki, numerator.shape)
    ru = ordered_product(cumulative_product_rows(factors), u)
    out = dif.at[:, :k].set(ordered_product(dif[:, :k], ru[:k, :k]))
    hinv = absh * invga
    matrix = mass - hinv * jac
    scale = 1 / jnp.max(jnp.abs(matrix), axis=1) if dae else jnp.ones(matrix.shape[0])
    return dict(dif=out, difRU=ru, hinvGak=hinv, Miter=scale[:, None] * matrix, RowScale=scale)
