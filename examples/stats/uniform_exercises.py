"""Uniform distribution exercises.

Uniform densities built as Chebfuns; means, quantiles and conditional
probabilities via sum/cumsum/roots, and a parameter-matching problem solved
with bivariate chebfun2 rootfinding.  Faithful port of stats/UniformExercises.m.

Original: https://www.chebfun.org/examples/stats/UniformExercises.html
Author: Jie Gao, July 2013
"""

import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.plotting import chebfun_style

chebfun_style()


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    # --- Problem 1(c): uniform on [1, 2] --------------------------------
    # MATLAB: f = chebfun(@(x) 1/(2-1)+0*x, [1 2]); fint = cumsum(f);
    #   mu_x = sum(chebfun(@(x) x.*f(x), [1 2]))
    #   a = roots(1-fint-1/4);  z = a-mu_x
    f = cj.chebfun(lambda x: 1.0 / (2 - 1) + 0 * x, domain=(1.0, 2.0))
    fint = f.cumsum()
    mu_x = float((cj.chebfun(lambda x: x, domain=(1.0, 2.0)) * f).sum())
    print(f"mu_x = {mu_x:.15f}")
    a = float(np.asarray((1 - fint - 0.25).roots()).ravel()[0])
    print(f"a = {a:.15f}")
    print(f"z = {a - mu_x:.15f}")

    # --- Problem 1(m): match mean=1, var=4/3 for a uniform on [a, b] -----
    # MATLAB: f = chebfun2(@(a,b) (a+b)/2 - 1, ...); g = chebfun2(@(a,b) (b-a)^2/12 - 4/3, ...);
    #   r = roots(f, g); a = min(r(2,:)); b = max(r(2,:))
    F = chebfun2(lambda a, b: (a + b) / 2 - 1.0, domain=(-5.0, 5.0, -5.0, 5.0))
    G = chebfun2(lambda a, b: (b - a) ** 2 / 12 - 4.0 / 3.0,
                 domain=(-5.0, 5.0, -5.0, 5.0))
    r = np.asarray(F.roots(G), dtype=float)
    print("r =")
    for row in r:
        print("   " + "   ".join(f"{v:.15f}" for v in row))
    a_m = float(np.min(r[1, :]))
    b_m = float(np.max(r[1, :]))
    print(f"a = {a_m:.15f}")
    print(f"b = {b_m:.15f}")

    # p = P(X < 0) for uniform on [a, b]
    fm = cj.chebfun(lambda x: 0 * x + 1.0 / (b_m - a_m), domain=(a_m, b_m))
    fmint = fm.cumsum()
    print(f"p = {float(fmint(jnp.array(0.0))):.15f}")

    # Single-variable form: b(a) = 2-a, aa = roots((b(a)-a)^2/12 - 4/3)
    g1 = cj.chebfun(lambda a: ((2 - a) - a) ** 2 / 12 - 4.0 / 3.0,
                    domain=(-5.0, 5.0))
    aa = np.sort(np.asarray(g1.roots()).ravel())
    print("aa =")
    for v in aa:
        print(f"   {v:.15f}")
    a_s = float(aa[0])
    print(f"a = {a_s:.15f}")
    print(f"b = {2 - a_s:.15f}")

    # Bivariate form again: ab = roots(meanab-1, varab-4/3); a=ab(1,2); b=ab(1,1)
    ab = np.asarray(F.roots(G), dtype=float)
    print("ab =")
    for row in ab:
        print("   " + "   ".join(f"{v:.15f}" for v in row))
    print(f"a = {float(ab[0, 1]):.15f}")
    print(f"b = {float(ab[0, 0]):.15f}")

    # --- Application: lottery wheel, uniform on [0, 360] -----------------
    fl = cj.chebfun(lambda x: 1.0 / (360 - 0) + 0 * x, domain=(0.0, 360.0))
    flint = fl.cumsum()
    p1 = float(flint(jnp.array(5.0 + 15.0)))
    print(f"p1 = {p1:.15f}")
    print(f"p1_exact = {(5 + 15) / 360:.15f}")
    pnb = float(1 - flint(jnp.array(80.0)))
    print(f"pnb = {pnb:.15f}")
    pnyb = float(1 - flint(jnp.array(35.0)) - flint(jnp.array(110.0)))
    print(f"pnyb = {pnyb:.15f}")
    pn = pnyb - float(flint(jnp.array(80.0)))
    print(f"pn = {pn:.15f}")
    p2 = pn / pnb
    print(f"p2 = {p2:.15f}")
    print(f"p2_exact = {(1 - (35 + 110 + 80) / 360) / (1 - 80 / 360):.15f}")

    # --- Plot: the lottery-wheel density with coloured sectors ----------
    fig, ax = plt.subplots()
    xs = np.linspace(0, 360, 400)
    ax.plot(xs, np.asarray(fl(jnp.array(xs))), 'k', lw=2)
    ax.set_title('Lottery wheel: uniform density on [0, 360]', fontsize=11)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'uniform_exercises.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    print("uniform_exercises: done")
    return True


if __name__ == "__main__":
    run()
