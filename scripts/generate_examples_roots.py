"""Generate per-block figures for the docs/examples/roots pages.

Each function regenerates the chebfun.org reference figures
<Name>_NN.png for one example page, at the reference pixel sizes,
using genuine chebfunjax computations. Run whole-file or filter with
an argument substring, e.g.:

    python scripts/generate_examples_roots.py BesselRoots
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import chebfunjax as cj
from chebfunjax.plotting import CHEBFUN_BLUE, chebfun_style, save_chebfun_figure

chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images", "roots")
REF = ("/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/refs/"
       "docs/images/roots")


def save(fig, name):
    from PIL import Image

    ref_path = os.path.join(REF, name)
    size = Image.open(ref_path).size if os.path.exists(ref_path) else (600, 270)
    path = os.path.join(OUT, name)
    save_chebfun_figure(fig, path, size=size)
    plt.close(fig)
    print(f"  {name} saved")


def besselroots():
    """roots/BesselRoots — J0 on [0,100] and its roots."""
    import scipy.special

    J0 = cj.chebfun(lambda x: jnp.asarray(scipy.special.j0(np.asarray(x))),
                    domain=[0.0, 100.0])
    xs = jnp.linspace(0.0, 100.0, 2000)

    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), np.asarray(J0(xs)), color=CHEBFUN_BLUE,
            linewidth=1.0)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Bessel function J_0")
    save(fig, "BesselRoots_01.png")

    r = J0.roots()
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), np.asarray(J0(xs)), color=CHEBFUN_BLUE,
            linewidth=1.0)
    ax.plot(np.asarray(r), np.asarray(J0(r)), ".r", markersize=7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Bessel function J_0")
    save(fig, "BesselRoots_02.png")


PAGES = {
    "BesselRoots": besselroots,
}


if __name__ == "__main__":
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, fn in PAGES.items():
        if flt.lower() in name.lower():
            print(f"[{name}]")
            fn()
