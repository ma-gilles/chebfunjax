"""Nonsmoothness of the field of values boundary.

Translation of linalg/NonsmoothFOV.m by Nick Trefethen
(July 2019): how smooth is the boundary curve of the field of
values?  Chebyshev/Fourier coefficient decay, AAA pole clustering,
and derivative jumps reveal near-corners, both with respect to the
Johnson angle and the true boundary angle.

The first matrix replays one captured MATLAB primitive input fixture; this does not claim general RNG parity.
Peter Maxwell's 5x5 complex matrix is hardcoded in the example.

Original: https://www.chebfun.org/examples/linalg/NonsmoothFOV.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import hashlib
import json
import os
import struct
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np  # uses-numpy: Matplotlib host-array interop only

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.chebfun1d.fov import fov  # noqa: E402
from chebfunjax.plotting import (
    chebfun_style,
    curve_plot_data,
    matlab_axis_equal,
    matlab_plot,
    plotregion,
    trig_plot_data,
)
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.aaa import aaa

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, "..", "..", "docs", "images", "linalg")

FIG = [0]
_RNG_FIXTURE_PATH = os.path.join(_HERE, "nonsmooth_fov_matlab_rng_2025b.json")
_MATLAB_RAW_BITS_SHA256 = "637e2710e954c22ed304c1afe001a257b1d3faae7da3a36564468187f3d4ecb8"
_MATLAB_FIXTURE_JSON_SHA256 = "bbf3d55e0472520aaf81ecc357b9a6cf89aafca613542fa4896e60a16946a660"


def _load_matlab_primitive_input(fixture_path):
    with open(fixture_path, "rb") as stream:
        payload_bytes = stream.read()
    if hashlib.sha256(payload_bytes).hexdigest() != _MATLAB_FIXTURE_JSON_SHA256:
        raise ValueError("MATLAB primitive fixture JSON hash mismatch")
    payload = json.loads(payload_bytes)
    if (
        payload.get("scope") != "MATLAB primitive RNG input replay only"
        or payload.get("source_expression") != "rng(1); n=60; Z=randn(n)"
        or payload.get("shape") != [60, 60]
        or payload.get("order") != "column-major"
        or payload.get("dtype") != "binary64"
        or payload.get("release") != "2025b"
        or payload.get("normal_transform") != "Ziggurat"
        or payload.get("repeat_bits_equal") is not True
    ):
        raise ValueError("unexpected MATLAB primitive fixture metadata")
    words = payload.get("hex")
    if not isinstance(words, list) or len(words) != 3600:
        raise ValueError("expected exactly 3600 captured MATLAB normal values")
    raw = b"".join(bytes.fromhex(word) for word in words)
    native_bits = b"".join(raw[i:i + 8][::-1] for i in range(0, len(raw), 8))
    if hashlib.sha256(native_bits).hexdigest() != _MATLAB_RAW_BITS_SHA256:
        raise ValueError("MATLAB raw primitive bit hash mismatch")
    values = struct.unpack(">3600d", raw)
    # MATLAB linear indexing is column-major; reshape in row-major then transpose.
    z = jnp.reshape(jnp.asarray(values, dtype=jnp.float64), (60, 60)).T
    return z / jnp.sqrt(jnp.asarray(60.0, dtype=jnp.float64))


def _source_style(fig, slot):
    from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
    ax = fig.axes[0]
    ax.set_position([.13, .16, .775, .75])
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(True, color=".85", linewidth=.5)
    ax.title.set_fontsize(11)
    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_fontsize(10)
    coeffs = slot in (2, 3, 7, 10, 11, 14)
    if coeffs:
        for line in ax.lines:
            line.set_markersize(1.2 if slot == 14 else .8 if slot in (10, 11) else 1.5)
            if slot in (10, 11, 14):
                line.set_markeredgewidth(.5 if slot == 14 else .4)
        lo = -15 if slot in (7, 14) else -20
        ax.set_ylim(10.**lo, 1)
        ax.set_yticks([1e-20, 1e-10, 1] if slot in (2, 3) else [10.**k for k in range(lo, 1, 5)])
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(1,), numticks=30))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="minor", color=".85", linewidth=.5, linestyle=":")
        limits = {2: (0, 900), 3: (-280, 280), 7: (-125, 125),
                  10: (0, 6000), 11: (-1900, 1900), 14: (-500, 500)}
        ax.set_xlim(*limits[slot])
        if slot == 14:
            ax.set_xticks(range(-400, 401, 100))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        if slot == 9:
            ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        for line in ax.lines:
            if line.get_marker() in (".", "o"):
                line.set_markersize(3)
        if slot in (1, 9):
            ax.set_position([.13, .11, .775, .795 if slot == 1 else .815])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim((-3, 3) if slot == 1 else (-2, 2))
            ax.set_ylim((-1.35, 1.35) if slot == 1 else (-.7, 1.05))
            ax.set_yticks([-1, -.5, 0, .5, 1] if slot == 1 else [-.5, 0, .5, 1])
        elif slot == 4:
            ax.set_position([.13, .11, .775, .815])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-.1 * np.pi, 2.1 * np.pi)
            ax.set_ylim(-1.6, 1.6)
            ax.set_xticks(range(7))
        else:
            ax.set_xlim(0, 2 * np.pi)
            ax.set_xticks(range(7))
            if slot in (5, 8):
                ax.set_ylim(-.3, .3)
                ax.set_yticks([-.3, -.2, -.1, 0, .1, .2, .3])
            elif slot == 6:
                ax.set_ylim(0, 2 * np.pi)
            elif slot == 12:
                ax.set_ylim(-4, 8)
                ax.set_yticks([-4, -2, 0, 2, 4, 6, 8])
            elif slot == 13:
                ax.set_ylim(0, 8)
                ax.set_yticks([0, 2, 4, 6, 8])


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    # Layout against the final 600x253 canvas; the helper applies its +1e-6 px
    # rounding guard again when it writes the PNG.
    fig.set_size_inches(6.0, 2.53)
    fig.axes[0].set_position([.13, .15, .775, .76])
    _source_style(fig, FIG[0])
    _savefig(fig, os.path.join(
        _IMG, f"NonsmoothFOV_{FIG[0]:02d}.png"), size=(600, 253))
    plt.close(fig)



def _plotcoeffs(c, title=None, fourier=False):
    fig, ax = plt.subplots(figsize=(6, 2.53))
    cj.plotcoeffs(
        c,
        ax=ax,
        title=title,
        source=True,
    )
    _save(fig)


def _curve_plot(A, c, xlim, title="field of values and eigenvalues"):
    fig, ax = plt.subplots(figsize=(6, 2.53))
    data = curve_plot_data(c)
    ax.plot(np.asarray(data["xLine"]), np.asarray(data["yLine"]), lw=2)
    w = jnp.linalg.eigvals(jnp.asarray(A))
    w_host = np.asarray(w)  # Matplotlib host-array boundary
    ax.plot(w_host.real, w_host.imag, '.k', ms=8)
    ax.set_position([.13, .15, .775, .76])
    ax.set_xlim(*xlim)
    matlab_axis_equal(ax)
    ax.grid(True)
    ax.set_title(title, fontsize=11)
    _save(fig)


def _real_plot(ax, f):
    from chebfunjax.tech.trigtech import Trigtech
    if all(isinstance(piece.tech, Trigtech) for piece in f.funs):
        data = trig_plot_data(f)
        ax.plot(np.asarray(data["xLine"]), np.asarray(data["yLine"]), lw=1.2)
        ax.set_xlim(float(f.domain.breakpoints[0]), float(f.domain.breakpoints[-1]))
    else:
        matlab_plot(f, ax=ax, linewidth=1.2)


def _analyze(A, xlim, second=False):
    c = fov(A)                      # c = fov(A)
    _curve_plot(A, c, xlim, title="" if second else "field of values and eigenvalues")
    print("ans =")
    print(f"   {len(c)}")
    _plotcoeffs(c)

    c_before_trig = c
    ct = cj.chebfun(lambda t: jnp.asarray(c_before_trig(t)),
                    domain=(0.0, 2 * jnp.pi), trig=True)
    print("ans =")
    print(f"   {len(ct)}")
    _plotcoeffs(
        ct, "Fourier coefficients wrt Johnson angle t" if not second else None,
        fourier=True,
    )
    c = ct

    xx = jnp.linspace(0, 2 * jnp.pi, 1000)
    if not second:
        fig, ax = plt.subplots(figsize=(6, 2.53))
        plotregion(c, ax=ax, title="")
        ax.set_position([.13, .15, .775, .76])
        matlab_axis_equal(ax)
        ax.grid(True)
        _, poles, *_ = aaa(c(xx), xx, tol=1e-10)
        poles = np.asarray(poles)
        ax.plot(poles.real, poles.imag, '.r', ms=6)
        ax.set_ylim(-1.6, 1.6)
        _save(fig)

    ac = c.abs()
    dac = ac.diff()
    fig, ax = plt.subplots(figsize=(6, 2.53))
    _real_plot(ax, dac)
    ax.grid(True)
    ax.set_xlabel("a" if second else "t")
    ax.set_ylabel("abs(d(a))'" if second else "abs(c(t))'")
    if second:
        ax.set_title("derivative of abs(d) wrt to true angle a",
                     fontsize=11)
    else:
        ax.set_title("derivative of abs(c) wrt to Johnson angle t",
                     fontsize=11)
    _save(fig)

    # a = 2*pi + unwrap(angle(c)); t = inv(a); d = chebfun(@(s) c(t(s)),[0 2*pi],'trig')
    a = 2 * jnp.pi + c.angle().unwrap()
    if second:
        old_a = a
        a = cj.chebfun(lambda s: old_a(s), domain=(0.0, 2 * jnp.pi))
    fig, ax = plt.subplots(figsize=(6, 2.53))
    _real_plot(ax, a)
    ax.grid(True)
    ax.set_xlabel("Johnson angle t")
    ax.set_ylabel("true angle a")
    _save(fig)
    t_of_a = a.inv()
    if second:
        t_domain = t_of_a.domain.breakpoints
        dfun = cj.chebfun(
            lambda s: c(t_of_a(s)),
            domain=(t_domain[0], t_domain[-1]),
            trig=True,
        )
        _plotcoeffs(dfun, fourier=True)
        return
    dfun = cj.chebfun(lambda s: c(t_of_a(s)), domain=(0.0, 2 * jnp.pi),
                      trig=True)
    _plotcoeffs(dfun, "Fourier coefficients wrt true angle a",
                fourier=True)

    add = dfun.abs().diff()
    fig, ax = plt.subplots(figsize=(6, 2.53))
    _real_plot(ax, add)
    ax.grid(True)
    ax.set_xlabel("a")
    ax.set_ylabel("abs(d(a))'")
    ax.set_title("derivative of abs(d) wrt to true angle a",
                 fontsize=11)
    _save(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    A = _load_matlab_primitive_input(_RNG_FIXTURE_PATH)
    _analyze(A, (-3, 3))

    A5 = jnp.asarray([
        [0.2560 + 0.0573j, 0.0568 + 0.0800j, 0.1597 + 0.2204j,
         -0.1649 + 0.1315j, -0.3639 + 0.0091j],
        [0.4733 + 0.2805j, -0.3192 + 0.1267j, 0.0810 + 0.0687j,
         0.5213 + 0.1574j, -0.0596 + 0.2879j],
        [0.1447 + 0.3037j, 0.2942 + 0.1844j, -0.2918 + 0.0364j,
         -0.2714 + 0.0265j, -0.0849 + 0.2264j],
        [-0.0650 + 0.1360j, 0.0952 + 0.0813j, -0.0503 + 0.0920j,
         -0.1500 + 0.0814j, 0.4742 + 0.1514j],
        [0.1938 + 0.0344j, 0.0419 + 0.1868j, -0.0453 + 0.0988j,
         -0.2207 + 0.2483j, -0.0772 + 0.1793j]], dtype=jnp.complex128)
    _analyze(A5, (-2, 2), second=True)


if __name__ == "__main__":
    run()
