"""The Gibbs phenomenon in 2D — original100x100 source computation.

Andre Uschmajew and Nick Trefethen, February2017.
Original: https://www.chebfun.org/examples/approx2/Gibbs2D.html
Copyright The University of Oxford and The Chebfun Developers.

CPU-qualified source candidate using public construction/extrema routes.
Native active-set, camlight/face interpolation and automatic
contour levels remain gaps; public renderer approximations are not parity.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.plotting import chebfun_style, matlab_view, save_chebfun_figure, spy

REFERENCE_SIZE = (600, 269)


def _answer(value, *, integer=False):
    print("ans =", flush=True)
    print(f"{int(value):6d}" if integer else f"  {float(value): .15f}", flush=True)


def _surface(fn):
    # Public source200x200 grid, no example-local mesh or camera-angle substitute.
    fig, ax = fn.plot()
    ax.set_zlim(-.2, 1.5)
    matlab_view(ax, -20, 50)
    # Native camlight sequence is bound in SOURCE_MAP; unsupported by renderer.
    return fig


def _contour(fn):
    # Public200x200 default grid. Native automatic-level policy is still absent;
    # do not insert cached/image-fitted levels to conceal that dependency.
    fig, ax = fn.contour(colorbar=True)
    ax.set_xlim(-.6, .6)
    ax.set_ylim(-.6, .6)
    ax.set_box_aspect(1)
    return fig


def run(output_dir=None):
    chebfun_style()
    output = Path(output_dir) if output_dir else Path(__file__).resolve().parents[2] / "docs/images/approx2"
    output.mkdir(parents=True, exist_ok=True)
    slot = 0

    def save(fig, *, layout=None):
        nonlocal slot
        slot += 1
        save_chebfun_figure(fig, output / f"Gibbs2D_{slot:02d}.png", size=REFERENCE_SIZE, layout=layout)
        plt.close(fig)

    A = jnp.zeros((100, 100), dtype=jnp.float64).at[39:61, 39:61].set(1)
    p = Chebfun2.from_values(A)
    save(_surface(p))  # Native camlight left, camlight left.
    save(_contour(p))
    max_p, _ = p.max2()
    _answer(max_p)

    a = jnp.zeros((100,), dtype=jnp.float64).at[39:61].set(1)
    p1 = chebfun(a)
    _, max_p1 = p1.max()
    _answer(max_p1)

    pzoom = p.restrict((0, .5, 0, .5))
    save(_surface(pzoom))  # Native camlight left.
    min_p, _ = p.min2()
    _answer(min_p)

    t = Chebfun2.from_values(A, trig=True)
    save(_surface(t))  # Native camlight, camlight, snapnow.
    save(_contour(t))
    max_t, _ = t.max2()
    _answer(max_t)
    min_t, _ = t.min2()
    _answer(min_t)

    A2 = jnp.tril(A)
    p2 = Chebfun2.from_values(A2)
    fig = _surface(p2.restrict((-.5, .5, -.5, .5)))  # Native camlight left.
    max_p2, _ = p2.max2()
    _answer(max_p2)
    min_p2, _ = p2.min2()
    _answer(min_p2)
    save(fig)  # Native snapnow occurs after both extrema calls.
    save(_contour(p2))

    _answer(p.rank, integer=True)
    _answer(t.rank, integer=True)
    _answer(p2.rank, integer=True)
    _answer(jnp.linalg.matrix_rank(A2), integer=True)

    fig, ax = plt.subplots(figsize=((600 + 1e-6) / 100, (269 + 1e-6) / 100), dpi=100)
    fig, ax = spy(A2, ax=ax)
    ax.set_xlim(36, 65)
    ax.set_ylim(65, 36)  # Native spy YDir reverse remains after axis([...]).
    save(fig, layout="matlab")
    assert slot == 8


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args.output_dir)
