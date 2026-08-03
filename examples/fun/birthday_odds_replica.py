"""Birthday odds.

Faithful replica of fun/BirthdayOdds.m by Jared Aurentz
(October 2014): the birthday-paradox probability as a chebfun of a
continuous party size, with the classic thresholds 23 and 57.

Original: https://www.chebfun.org/examples/fun/BirthdayOdds.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'fun')


def match_prob(n):
    nr = int(np.floor(n))
    a = np.sum(np.log(np.arange(1, 366)))
    if nr == n:
        k = int(n)
        return 1 - np.exp(a - np.sum(np.log(np.arange(
            1, 366 - k))) - k * np.log(365))
    v1 = np.prod(np.arange(365, 365 - nr, -1) / 365.0)
    v2 = np.prod(np.arange(365 - nr, 1, -1)
                 / (365 - n - np.arange(0, 364 - nr)))
    return 1 - v1 * v2 / gamma(2 - n + nr) / 365**(n - nr)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    ks = np.arange(1, 366)
    a = np.sum(np.log(np.arange(1, 366)))
    pk = np.array([1 - np.exp(a - np.sum(np.log(np.arange(
        1, 366 - k))) - k * np.log(365)) for k in ks])
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.plot(ks, pk, '.b', ms=4)
    ax.axis([0, 365, -0.1, 1.1])
    ax.set_xlabel("Size of the party")
    ax.set_ylabel("Probability of finding at least one pair")

    def op(narr):
        narr = np.atleast_1d(np.asarray(narr, dtype=float))
        return jnp.asarray([match_prob(v) for v in narr.ravel()]
                           ).reshape(np.shape(narr))

    p = cj.chebfun(op, domain=(1.0, 365.0))
    ns = np.linspace(1, 365, 700)
    ax.plot(ns, np.asarray(p(ns)), 'r', lw=1.2)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "BirthdayOdds_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    r50 = np.ceil(np.asarray((p - 0.5).roots()))
    print("ans =")
    print(f"    {int(r50[0])}")
    r99 = np.ceil(np.asarray((p - 0.99).roots()))
    print("ans =")
    print(f"    {int(r99[0])}")


if __name__ == "__main__":
    run()
