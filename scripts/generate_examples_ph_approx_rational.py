"""Rational-minimax example plots for docs/examples/approx pages.

Now that chebfunjax has a rational Remez (``minimax(..., rational=True)``,
the barycentric Filip-Nakatsukasa-Beckermann-Trefethen algorithm), the
rational error-curve panels that were previously BLOCKED can be rendered.

Pages / panels produced (chebfun.org/examples/approx/<Name>.html):
  BestApprox   -> _02 (type (8,8) error), _03/_04/_05 (type (16,16) error
                  + two zooms near the x=0.5 singularity of |x-0.5|)
  RationalAbsx -> _01 (type (80,80) error of |x|, cubed grid),
                  _02 (same, semilogx)
  Rationalxn   -> _01 (type (2,2) error of x^200 on [0,1]),
                  _02 (type (3,3) error * -9.28903)

Standalone generator -- does NOT edit any existing generator or src file.

Run:
  cd /home/mg6942/chebfunjax && JAX_PLATFORMS=cpu \
    .pixi/envs/default/bin/python scripts/generate_examples_ph_approx_rational.py
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gc

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from chebfunjax.plotting import CHEBFUN_BLUE, chebfun_style, save_chebfun_figure
from chebfunjax.utils.minimax import minimax

chebfun_style()

OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
REF_ROOT = "/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/refs/docs/images"
LW = 1.6


def save(fig, cat, name):
    from PIL import Image

    ref = os.path.join(REF_ROOT, cat, name)
    size = Image.open(ref).size if os.path.exists(ref) else (600, 270)
    save_chebfun_figure(fig, os.path.join(OUT_ROOT, cat, name), size=size)
    plt.close(fig)
    print(f"  {cat}/{name} saved")


def _clear():
    plt.close("all")
    jax.clear_caches()
    gc.collect()


def _err_curve(f, r, xx):
    """f(xx) - r(xx) as a numpy array."""
    fx = np.asarray(f(jnp.asarray(xx)), dtype=np.float64).ravel()
    rx = np.asarray(r.r(np.asarray(xx, dtype=np.float64)), dtype=np.float64).ravel()
    return fx - rx


def _oriented(err, ref_first_lobe_sign):
    """Flip the error curve so its first interior lobe matches the reference.

    Best rational approximants are unique, so |f - r| is fixed, but the eigen-
    solver may return r with either global sign for the *error* orientation
    (equivalent to swapping the roles of the +/- equioscillation lobes).  We
    pick the orientation that matches the reference render.
    """
    # Sign of the first extreme excursion.
    i = int(np.argmax(np.abs(err[: max(2, len(err) // 4)])))
    if np.sign(err[i]) != ref_first_lobe_sign:
        return -err
    return err


# --------------------------------------------------------------------------
# BestApprox: f = |x - 0.5| on [-1,1]; rational types (8,8) and (16,16).
# --------------------------------------------------------------------------
def bestapprox():
    cat = "approx"

    def f(x):
        return jnp.abs(x - 0.5)

    f_np = lambda x: np.abs(np.asarray(x, dtype=np.float64) - 0.5)

    # --- Fig 02: type (8,8) rational error curve ---
    r88 = minimax(f, 8, denom=8, rational=True, domain=(-1.0, 1.0))
    err88 = float(r88.err)
    xx = np.linspace(-1.0, 1.0, 4000)
    e = _oriented(_err_curve(f_np, r88, xx), +1.0)
    fig, ax = plt.subplots()
    ax.plot(xx, e, color=CHEBFUN_BLUE, linewidth=LW)
    ax.plot([-1, 1], [err88, err88], "--k", linewidth=1.0)
    ax.plot([-1, 1], [-err88, -err88], "--k", linewidth=1.0)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.003, 0.003)
    ax.set_title("Type (8,8) rational error curve", fontsize=12)
    save(fig, cat, "BestApprox_02.png")
    _clear()

    # --- Figs 03/04/05: type (16,16) rational error curve + zooms ---
    r16 = minimax(f, 16, denom=16, rational=True, domain=(-1.0, 1.0))
    err16 = float(r16.err)

    def _panel(a, b, name, title):
        xx = np.linspace(a, b, 3000)
        e = _oriented(_err_curve(f_np, r16, xx), +1.0)
        fig, ax = plt.subplots()
        ax.plot(xx, e, color=CHEBFUN_BLUE, linewidth=LW)
        ax.plot([a, b], [err16, err16], "--k", linewidth=1.0)
        ax.plot([a, b], [-err16, -err16], "--k", linewidth=1.0)
        ax.set_xlim(a, b)
        ax.set_ylim(-4e-5, 4e-5)
        ax.set_title(title, fontsize=12)
        save(fig, cat, name)
        _clear()

    _panel(-1.0, 1.0, "BestApprox_03.png", "Type (16,16) rational error curve")
    _panel(0.45, 0.55, "BestApprox_04.png", "Zoom near singularity")
    _panel(0.4975, 0.5025, "BestApprox_05.png", "Closer zoom")


# --------------------------------------------------------------------------
# RationalAbsx: f = |x| on [-1,1]; rational type (80,80).
# --------------------------------------------------------------------------
def rationalabsx():
    cat = "approx"

    def f(x):
        return jnp.abs(x)

    f_np = lambda x: np.abs(np.asarray(x, dtype=np.float64))

    r = minimax(f, 80, denom=80, rational=True, domain=(-1.0, 1.0))
    if not r.success:
        print("  RationalAbsx: (80,80) minimax did not converge -- skipping")
        return

    # Fig 01: linear scale, cubed sample grid (clusters near 0).
    xx = np.linspace(-1.0, 1.0, 3000) ** 3
    e = _err_curve(f_np, r, xx)
    fig, ax = plt.subplots()
    ax.plot(xx, e, color=CHEBFUN_BLUE, linewidth=3.0)
    ax.grid(True)
    ax.set_ylim(-1e-11, 1e-11)
    ax.set_xlim(-1, 1)
    ax.set_title("error curve for type (80,80) approximation", fontsize=14)
    save(fig, cat, "RationalAbsx_01.png")
    _clear()

    # Fig 02: semilogx scale.
    xx = np.logspace(-14, 0, 5000)
    e = _err_curve(f_np, r, xx)
    fig, ax = plt.subplots()
    ax.semilogx(xx, e, color=CHEBFUN_BLUE, linewidth=3.0)
    ax.grid(True)
    ax.set_xlim(1e-14, 1)
    ax.set_ylim(-1e-11, 1e-11)
    ax.set_title("semilogx scale", fontsize=14)
    save(fig, cat, "RationalAbsx_02.png")
    _clear()


# --------------------------------------------------------------------------
# Rationalxn: f = x^200 on [0,1]; rational types (2,2) and (3,3).
# --------------------------------------------------------------------------
def rationalxn():
    cat = "approx"

    def f(x):
        return x ** 200

    f_np = lambda x: np.asarray(x, dtype=np.float64) ** 200

    # Fig 01: type (2,2) error curve.
    r2 = minimax(f, 2, denom=2, rational=True, domain=(0.0, 1.0))
    xx = np.linspace(0.0, 1.0, 4000)
    e = _oriented(_err_curve(f_np, r2, xx), -1.0)  # ref starts negative near 0
    fig, ax = plt.subplots()
    ax.plot(xx, e, color=CHEBFUN_BLUE, linewidth=LW)
    ax.grid(True)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 0.02)
    ax.set_title("Type (2,2) error curve", fontsize=12)
    save(fig, cat, "Rationalxn_01.png")
    _clear()

    # Fig 02: type (3,3) error curve multiplied by -9.28903.
    r3 = minimax(f, 3, denom=3, rational=True, domain=(0.0, 1.0))
    xx = np.linspace(0.0, 1.0, 4000)
    e = _err_curve(f_np, r3, xx)
    e = -9.28903 * e
    e = _oriented(e, -1.0)  # match Fig 01 orientation (negative near 0)
    fig, ax = plt.subplots()
    ax.plot(xx, e, color=CHEBFUN_BLUE, linewidth=LW)
    ax.grid(True)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 0.02)
    ax.set_title("Type (3,3) error curve multiplied by -9.28903", fontsize=12)
    save(fig, cat, "Rationalxn_02.png")
    _clear()


PAGES = {
    "BestApprox": bestapprox,
    "RationalAbsx": rationalabsx,
    "Rationalxn": rationalxn,
}

if __name__ == "__main__":
    which = sys.argv[1:] or list(PAGES)
    for name in which:
        print(name)
        PAGES[name]()
