"""Phase portraits of complex functions.

A phase portrait encodes a complex function f(z) by its argument (phase),
colouring each point z by arg(f(z)).  Zeros appear as points where all
colours meet, and poles as points where colours meet in the opposite orientation.

Credit: Inspired by Chebfun examples complex/PhasePortraits.m (Alex Townsend,
March 2013) and complex/PhaseplotCommand.m (Nick Trefethen, October 2020).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

def phase_plot(ax, f_complex, xlim, ylim, n=400, title=""):
    """Draw the phase portrait of f on a grid."""
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = XX + 1j * YY
    FZ = f_complex(ZZ)
    phase = np.angle(FZ)
    # Use a circular colormap (HSV)
    ax.pcolormesh(xs, ys, phase, cmap='hsv', shading='auto',
                  vmin=-np.pi, vmax=np.pi)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    if title:
        ax.set_title(title, fontsize=9)

def run():
    print("=" * 60)
    print("Phase portraits of complex functions")
    print("=" * 60)

    pi = float(jnp.pi)

    # --- Example 1: f(z) = z (identity) --------------------------------
    # Phase portrait shows a rainbow wheel centred at the origin
    # Verify that zeros/poles are at expected locations
    functions = [
        ("z", lambda z: z, (-2, 2), (-2, 2)),
        ("z^2", lambda z: z**2, (-2, 2), (-2, 2)),
        ("(z-1)/(z+1)", lambda z: (z - 1) / (z + 1), (-3, 3), (-3, 3)),
        ("sin(z)", lambda z: np.sin(z), (-pi, pi), (-3, 3)),
        ("exp(1/z)", lambda z: np.exp(1.0/z), (-2, 2), (-2, 2)),
        ("(z^2-1)*(z-2-1j)^2 / (z^2+2+2j)", lambda z: (z**2-1)*(z-2-1j)**2/(z**2+2+2j),
         (-3, 3), (-3, 3)),
    ]

    # Verify residue theorem for (z-1)/(z+1): pole at z=-1
    # int_{|z-(-1)|=0.5} (z-1)/(z+1) dz = 2*pi*i * Res[f, -1] = 2*pi*i * (-2)
    def f_rat(z):
        return (z - 1.0) / (z + 1.0)

    ts = np.linspace(0, 2*pi, 4000, endpoint=False)
    center = -1.0
    r = 0.5
    zs = center + r * np.exp(1j * ts)
    dzs = 1j * r * np.exp(1j * ts) * (2 * pi / 4000)
    fs = f_rat(zs)
    I = np.sum(fs * dzs)
    exact = 2j * pi * (-2.0)  # Residue at z=-1: lim_{z->-1} (z+1) * (z-1)/(z+1) = -2
    err = abs(I - exact)
    print(f"\n(z-1)/(z+1) residue at z=-1: computed = {I.real:.6f} + {I.imag:.6f}i")
    print(f"  Exact 2*pi*i*(-2) = {exact.real:.6f} + {exact.imag:.6f}i")
    print(f"  Error: {err:.2e}")
    assert err < 1e-6, f"Residue error {err}"

    # Verify zero at z=1 (phase winds clockwise around it)
    r_zero = 0.3
    zs_zero = 1.0 + r_zero * np.exp(1j * ts)
    fs_zero = f_rat(zs_zero)
    args_zero = np.unwrap(np.angle(fs_zero))
    winding = (args_zero[-1] - args_zero[0]) / (2 * pi)
    print(f"\nWinding number of (z-1)/(z+1) around z=1 (zero): {winding:.4f}")
    print(f"  (Expected +1: one zero)")
    assert abs(winding - 1.0) < 0.02, f"Winding number at zero: {winding}"

    # Verify pole at z=-1 (phase winds clockwise)
    zs_pole = -1.0 + r_zero * np.exp(1j * ts)
    fs_pole = f_rat(zs_pole)
    args_pole = np.unwrap(np.angle(fs_pole))
    winding_pole = (args_pole[-1] - args_pole[0]) / (2 * pi)
    print(f"Winding number around z=-1 (pole): {winding_pole:.4f}")
    print(f"  (Expected -1: one pole)")
    assert abs(winding_pole - (-1.0)) < 0.02, f"Winding at pole: {winding_pole}"

    # sin(z): zeros at n*pi
    def f_sin(z):
        return np.sin(z)

    for n in [-2, -1, 0, 1, 2]:
        z0 = n * pi
        r_test = 0.5
        zs_n = z0 + r_test * np.exp(1j * ts)
        fs_n = f_sin(zs_n)
        args_n = np.unwrap(np.angle(fs_n))
        w = (args_n[-1] - args_n[0]) / (2 * pi)
        print(f"sin(z) winding around z={n}*pi: {w:.3f}  (expected 1)")
        assert abs(w - 1.0) < 0.05, f"sin winding at {n}*pi: {w}"

    # --- Plots ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(2, 3)
    axes = axes.flatten()

    for i, (name, f_np, xlim, ylim) in enumerate(functions):
        phase_plot(axes[i], f_np, xlim, ylim, title=f"$f(z) = {name}$")

    fig.suptitle("Phase portraits: hue = arg f(z)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "phase_portraits.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
