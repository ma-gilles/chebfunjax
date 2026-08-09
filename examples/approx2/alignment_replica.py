"""Low-rank approximation and alignment with axes.

Faithful replica of approx2/Alignment.m (Trefethen, 2016): the
numerical rank of tanh(k*(c*x + s*y)) depends strongly on how the
function is aligned with the coordinate axes -- rank 1 when aligned,
r ~ 0.6*m at 45 degrees, so low-rank compression is no win there.

Original: https://www.chebfun.org/examples/approx2/Alignment.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx2')


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    k = 3
    f = Chebfun2.from_function(lambda x, y: np.tanh(k * x))
    print("r =")
    print(f"     {int(f.rank)}")

    m, n = f.length()
    print("m =")
    print(f"    {m}")
    print("n =")
    print(f"     {n}")

    f45 = Chebfun2.from_function(
        lambda x, y: np.tanh(k * (x + y) / np.sqrt(2)))
    print("r =")
    print(f"    {int(f45.rank)}")
    m, n = f45.length()
    print("m =")
    print(f"    {m}")
    print("n =")
    print(f"    {n}")

    print('    theta   rank     m     n')
    for theta in np.arange(0, 1.57 + 1e-12, .157):
        c, s = np.cos(theta), np.sin(theta)
        ft = Chebfun2.from_function(
            lambda x, y: np.tanh(k * (c * x + s * y)))
        m, n = ft.length()
        print(f"{theta:9.4f}  {int(ft.rank):5d} {m:5d} {n:5d}")

    tt = np.linspace(0, np.pi / 2, 100)
    rr = []
    for theta in tt:
        c, s = np.cos(theta), np.sin(theta)
        ft = Chebfun2.from_function(
            lambda x, y: np.tanh(3 * (c * x + s * y)))
        rr.append(int(ft.rank))
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(tt, rr, '.-')
    ax.grid(True)
    ax.set_xlabel('theta')
    ax.set_ylabel('rank r')
    ax.set_title('rank vs. angle')
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Alignment_repl_01.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    print('       k      r     m     n     r/m')
    for kk in range(1, 11):
        f45 = Chebfun2.from_function(
            lambda x, y: np.tanh(kk * (x + y) / np.sqrt(2)))
        r = int(f45.rank)
        m, n = f45.length()
        print(f"{kk:8.2f}  {r:5d} {m:5d} {n:5d} {r / m:7.2f}")


if __name__ == "__main__":
    run()
