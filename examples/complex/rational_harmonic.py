"""Zeros of rational harmonic functions with source Chebfun2 phase portraits.

Translation of complex/RationalHarmonic.m by Olivier Sete (February2016).
Original: https://www.chebfun.org/examples/complex/RationalHarmonic.html
Copyright by The University of Oxford and The Chebfun Developers.
"""

import matplotlib

matplotlib.use("Agg")
import hashlib
import json
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.phaseplot import phaseplot

chebfun_style()
_IMG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "images", "complex"
)
_PAGE_REPORT = None
N = 3
A = 0.7
DOM = (-1.4, 1.4, -1.4, 1.4)


def _common_roots(expr):
    # Explicit MATLAB roots(complexChebfun2) real/imaginary host-output adapter.
    fre = cj.chebfun2(lambda x, y: jnp.real(expr(x, y)), domain=DOM)
    fim = cj.chebfun2(lambda x, y: jnp.imag(expr(x, y)), domain=DOM)
    return np.asarray(fre.roots(fim)).reshape(-1, 2)


def smash(v):
    value = v / (1 + jnp.abs(v) ** 2)
    return jnp.where(jnp.isnan(value), 0, value)


def _portrait(fun, zeros, poles, index):
    # The source explicitly constructs the smooth Chebfun2 at eps1e-8.
    portrait = cj.chebfun2(lambda x, y: fun(x + 1j * y), domain=DOM, tol=1e-8)
    # classic hue with caxis_start0 equals angle(-f) with caxis[-pi,pi],
    # as in pinned @separableApprox/plot.m; existing HSV600 adapter reused.
    image = phaseplot(lambda z: portrait(jnp.real(z), jnp.imag(z)), ax=DOM, n_pts=500, classic=True)
    fig, ax = plt.subplots(figsize=(610 / 72.009, 276 / 72.009))
    ax.set_position([203 / 610, 30 / 276, 225 / 610, 225 / 276])
    ax.imshow(image, origin="lower", extent=DOM, aspect="equal")
    ax.set_xlim(DOM[:2])
    ax.set_ylim(DOM[2:])
    ax.set_axis_off()
    ax.plot(poles[:, 0], poles[:, 1], "ws", markersize=3 * 100 / 72.009, markerfacecolor="w")
    ax.plot(zeros[:, 0], zeros[:, 1], "ko", markersize=3 * 100 / 72.009, markerfacecolor="k")
    _savefig(
        fig, os.path.join(_IMG, f"RationalHarmonic_{index:02d}.png"), size=(610, 276), dpi=72.009
    )
    plt.close(fig)
    return {
        "rank": portrait.rank,
        "raster_shape": list(image.shape),
        "raster_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "tol": 1e-8,
    }


def run():
    os.makedirs(_IMG, exist_ok=True)

    def p(z):
        return z ** (N - 1)

    def q(z):
        return z**N - A**N

    poles = _common_roots(lambda x, y: q(x + 1j * y))
    zeros = _common_roots(lambda x, y: p(x + 1j * y) - q(x + 1j * y) * jnp.conj(x + 1j * y))

    def ff(z):
        return p(z) / q(z) - jnp.conj(z)

    report1 = _portrait(lambda z: smash(ff(z)), zeros, poles, 1)
    epsilon = 0.01
    poles_eps = _common_roots(lambda x, y: q(x + 1j * y) * (x + 1j * y))
    zeros_eps = _common_roots(
        lambda x, y: (
            p(x + 1j * y) * (x + 1j * y)
            + epsilon * q(x + 1j * y)
            - q(x + 1j * y) * (x + 1j * y) * jnp.conj(x + 1j * y)
        )
    )
    print("ans =")
    print(f"    {len(zeros_eps)}")
    report2 = _portrait(
        lambda z: smash((ff(z) + epsilon / z) * jnp.abs(z * q(z)) ** 2), zeros_eps, poles_eps, 2
    )
    if _PAGE_REPORT is not None:
        with open(_PAGE_REPORT, "w") as stream:
            json.dump(
                {
                    "pole_count": len(poles),
                    "zero_count": len(zeros),
                    "perturbed_pole_count": len(poles_eps),
                    "perturbed_zero_count": len(zeros_eps),
                    "portrait1": report1,
                    "portrait2": report2,
                    "roots": zeros.tolist(),
                    "perturbed_roots": zeros_eps.tolist(),
                },
                stream,
                indent=2,
            )


if __name__ == "__main__":
    run()
