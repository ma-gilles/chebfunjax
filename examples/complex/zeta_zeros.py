"""Zeros of zeta(s) by polynomial analytic continuation.

Provenance
----------
MATLAB source : complex/ZetaZeros.m, Nick Trefethen and Mohsin Javed,
July 2015. Reference Chebfun library commit: 7574c77.
The full descending 100000-term source sum is retained. JAX binary64
reduction/complex-power arithmetic can differ from MATLAB; no coefficient,
root-bit or pixel parity is claimed.
"""
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import chebfunjax as cj
from chebfunjax.plotting import (
    CHEBFUN_BLUE,
    chebfun_style,
    plotregion,
    save_chebfun_figure,
)

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.environ.get("ZETAZEROS_IMAGE_DIR") or os.path.join(
    _HERE, "..", "..", "docs", "images", "complex"
)


@jax.jit
def _zeta_scalar(s):
    terms = jnp.arange(100000, 0, -1, dtype=jnp.float64)
    return jnp.sum(terms ** (-jnp.asarray(s, dtype=jnp.complex128)))


def zeta(s):
    """Literal descending Dirichlet sum, scalar-vectorized as native source.

    Provenance
    ----------
    MATLAB source : complex/ZetaZeros.m, zeta handle and vectorize constructor.
    Chebfun commit: 7574c77 (reference library; example caller).
    """
    values = jnp.asarray(s, dtype=jnp.complex128)
    return jnp.stack([_zeta_scalar(value) for value in values.ravel()]).reshape(
        values.shape
    )


def run():
    """Execute all original ZetaZeros computation and display cells.

    Provenance
    ----------
    MATLAB source : complex/ZetaZeros.m, Nick Trefethen and Mohsin Javed.
    Chebfun commit: 7574c77 (reference library; example caller).
    """
    os.makedirs(_IMG, exist_ok=True)
    capture = os.environ.get("ZETAZEROS_CAPTURE_DIR")
    if capture:
        os.makedirs(capture, exist_ok=True)
    started = time.perf_counter()
    value = zeta(4.0)
    print("ans =")
    print(f"   {float(jnp.real(value)):.15f}")
    print("exact =")
    print(f"   {float(jnp.pi**4 / 90):.15f}")

    s = cj.chebfun(lambda t: 4.0 + 1j * t, domain=[5.0, 50.0])
    call_count = [0]

    def observed(t):
        arguments = s(t)
        values = zeta(arguments)
        if capture:
            call_count[0] += 1
            np.savez(os.path.join(capture, f"construction_{call_count[0]:02d}.npz"),
                     t=np.asarray(t), s=np.asarray(arguments), values=np.asarray(values))
        return values

    f = cj.chebfun(observed, domain=[5.0, 50.0])
    print("f =")
    print(repr(f))
    zeros_t = np.asarray(f.roots(complex_roots=True, norecursion=True))

    fig, ax = plt.subplots(figsize=(6, 2.7), dpi=100)
    plotregion(f, ax=ax, title="")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim(-5, 60)
    ax.set_aspect("equal", adjustable="datalim")
    ends = np.asarray(f.domain.breakpoints)
    ax.plot(ends, np.zeros_like(ends), "k+-")
    ax.plot(zeros_t.real, zeros_t.imag, ".r", markersize=3)
    ax.plot(0, 3, "xk", markersize=12)
    ax.set_yticks(range(-12, 13, 4))
    ax.grid(True)
    save_chebfun_figure(fig, os.path.join(_IMG, "ZetaZeros_01.png"), layout="matlab")
    plt.close(fig)

    zeros_s = np.asarray(s(jnp.asarray(zeros_t)))
    zeros_exact = 0.5 + 1j * np.asarray([
        14.1347251417, 21.0220396388, 25.0108575801, 30.4248761259,
        32.9350615877, 37.5861781588, 40.9187190121, 43.3270732809,
    ])
    if capture:
        np.savez(os.path.join(capture, "continuation.npz"),
                 s_coeffs=np.asarray(s.chebcoeffs()),
                 f_coeffs=np.asarray(f.chebcoeffs()), zeros_t=zeros_t,
                 zeros_s=zeros_s, zeros_exact=zeros_exact)
    # Source fprintf requires matching rows; never silently truncate/filter roots.
    if len(zeros_s) != len(zeros_exact):
        raise RuntimeError(f"Source table has {len(zeros_s)} computed roots, expected eight rows")
    print("            Chebfun                          Exact")
    for computed, exact in zip(zeros_s, zeros_exact, strict=True):
        print(f"{computed.real:13.10f} + {computed.imag:13.10f}i   "
              f"{exact.real:13.10f} + {exact.imag:13.10f}i")

    t = cj.chebfun(lambda x: 3.5j + x, domain=[5.0, 50.0])
    ft = f(t)
    imaginary = ft.imag()
    real = ft.real()
    fig, ax = plt.subplots(figsize=(6, 2.7), dpi=100)
    imaginary.plot(ax=ax, color=CHEBFUN_BLUE)
    real.plot(ax=ax, color="#D95319")  # Native second default ColorOrder entry.
    ax.set_title("Real and imaginary parts of zeta(s) along critical line", fontsize=10)
    ax.plot(zeros_t.real, (zeros_t - 3.5j).imag, ".k", markersize=3)
    ax.grid(True)
    save_chebfun_figure(fig, os.path.join(_IMG, "ZetaZeros_02.png"), layout="matlab")
    plt.close(fig)
    if capture:
        np.savez(os.path.join(capture, "critical_line.npz"),
                 t_coeffs=np.asarray(t.chebcoeffs()), ft_coeffs=np.asarray(ft.chebcoeffs()),
                 imaginary_coeffs=np.asarray(imaginary.chebcoeffs()),
                 real_coeffs=np.asarray(real.chebcoeffs()))
    print(f"Elapsed time is {time.perf_counter() - started:.6f} seconds.")


if __name__ == "__main__":
    run()
