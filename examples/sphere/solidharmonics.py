"""Solid harmonics: literal computations from sphere/SolidHarmonics.m.

Original: https://www.chebfun.org/examples/sphere/SolidHarmonics.html
Nicolas Boulle and Alex Townsend, May 2019.
Copyright by The University of Oxford and The Chebfun Developers.

Native prose and input cells are retained in docs/examples/sphere/SolidHarmonics.md.
Default public Ballfun plots use source cylindrical data. Native lighting and
face interpolation remain unqualified; elapsed time is Python execution only.
"""
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig


def run():
    chebfun_style()
    image_dir = Path(os.environ.get(
        "CHEBFUN_EXAMPLE_IMAGE_DIR",
        Path(__file__).resolve().parents[2] / "docs/images/sphere",
    ))
    image_dir.mkdir(parents=True, exist_ok=True)

    R42 = Ballfun.solharm(4, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    R42.plot(ax=ax)
    ax.set_axis_off()
    _savefig(fig, image_dir / "SolidHarmonics_01.png", size=(600, 253))
    plt.close(fig)

    print("ans =")
    print(f"     {float(R42.laplacian().norm()):.15e}")
    R40 = Ballfun.solharm(4, 0)
    # Public sum() without dimension is the documented native sum3 mapping.
    print("ans =")
    print(f"   {float((R42 * R42).sum()):.15f}")
    print("ans =")
    print(f"   {float((R40 * R40).sum()):.15f}")
    print("ans =")
    print(f"     {float((R42 * R40).sum()):.15e}")

    N = 3
    fig = plt.figure()
    for l in range(N + 1):
        for m in range(l + 1):
            R = Ballfun.solharm(l, m)
            ax = fig.add_subplot(N + 1, N + 1, l * (N + 1) + m + 1,
                                 projection="3d")
            R.plot(ax=ax)
            ax.set_axis_off()
    _savefig(fig, image_dir / "SolidHarmonics_02.png", size=(600, 253))
    plt.close(fig)

    start = time.perf_counter()
    high_degree = Ballfun.solharm(150, 50)
    high_degree.coeffs.block_until_ready()
    print(f"Elapsed time is {time.perf_counter() - start:.6f} seconds.")


if __name__ == "__main__":
    run()
