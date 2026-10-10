"""Needle on a corrugated surface.

Translation of opt/Needle.m by Nick Trefethen and Hrothgar (December 2013).
Original: https://www.chebfun.org/examples/opt/Needle.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import os
import sys
import time

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax import chebfun
from chebfunjax.plotting import PARULA, chebfun_style, matlab_plot, save_chebfun_figure
from chebfunjax.utils._fminsearch import fminsearch
from chebfunjax.utils._interp2_cubic import uniform_cubic_interp2
from chebfunjax.utils._matlab_linspace import source_grid

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'opt')


def run():
    chebfun_style()
    os.makedirs(_IMG, exist_ok=True)
    s = chebfun('s', domain=[-4., 4.])
    h = .1*s**2 + .1*(6*s).sin() + .03*(12*s).sin()
    figure_number = 0

    def save(fig):
        nonlocal figure_number
        figure_number += 1
        if figure_number in (2, 3, 4):
            fig.set_size_inches(6, 2.7)
            for axis in fig.axes:
                axis.tick_params(labelsize=9)
            fig.tight_layout(pad=.6, h_pad=1.2)
        save_chebfun_figure(
            fig, os.path.join(_IMG, f'Needle_{figure_number:02d}.png'),
            size=(600, 270))
        plt.close(fig)

    def minfun(x, theta):
        r = float(.5*jnp.cos(theta))
        # MATLAB brace subsref restricts, then simplifies the result.
        hx = h.restrict(x-r, x+r).simplify()
        needle = chebfun(lambda s: jnp.tan(theta)*(s-x), domain=[x-r, x+r])
        _, value = (hx-needle).max()
        return value

    def plotneedle(ax, x, theta):
        y = minfun(x, theta)
        r = float(.5*jnp.cos(theta))
        h.restrict(x-r, x+r).simplify()  # The source also evaluates this brace call.
        needle = chebfun(lambda s: y+jnp.tan(theta)*(s-x), domain=[x-r, x+r])
        matlab_plot(h, 'b', needle, 'k', ax=ax, linewidth=1)
        ax.set_aspect('equal')
        ax.axis([-4, 4, -.4, 2])

    fig, ax = plt.subplots(figsize=(6, 2.7))
    matlab_plot(h, ax=ax, linewidth=1)
    ax.set_aspect('equal')
    ax.axis([-4, 4, -.4, 2])
    save(fig)

    fig, axes = plt.subplots(2, 1, figsize=(6, 2.7))
    plotneedle(axes[0], -.6, -.2)
    axes[0].set_title('needle with (x,theta) = (-0.6, -0.2)', fontsize=10, fontweight="bold")
    plotneedle(axes[1], 1.7, 1.)
    axes[1].set_title('needle with (x,theta) = (1.7, 1)', fontsize=10, fontweight="bold")
    for axis in axes:
        axis.set_yticks([0., .5, 1., 1.5, 2.])
    save(fig)

    def contour_grid(x_domain, theta_domain, levels, colorbar):
        start = time.perf_counter()
        x = source_grid(*x_domain, 25)
        theta = source_grid(*theta_domain, 25)
        yy = jnp.zeros((25, 25), dtype=jnp.float64)
        # Preserve native loop order: columns (x), then rows (theta).
        for k in range(len(x)):
            for j in range(len(theta)):
                yy = yy.at[j, k].set(minfun(float(x[k]), float(theta[j])))
        xxp = source_grid(*x_domain, 100)
        ttp = source_grid(*theta_domain, 100)
        yyp = uniform_cubic_interp2(
            yy, xxp[None, :], ttp[:, None],
            x_domain=x_domain, y_domain=theta_domain)
        fig, ax = plt.subplots(figsize=(6, 2.7))
        contours = ax.contour(xxp, ttp, yyp, levels=levels, cmap=PARULA, linewidths=.5)
        ax.grid(True)
        ax.set_xlabel('x', fontsize=9)
        ax.set_ylabel('theta', fontsize=9)
        if colorbar:
            fig.colorbar(contours, ax=ax)
        ax.set_title(f'min value on grid: {float(jnp.min(yy)):.5g}', fontsize=10, fontweight="bold")
        elapsed = time.perf_counter()-start
        print(f'Elapsed time is {elapsed:.6f} seconds.', flush=True)
        save(fig)

    contour_grid((-2., 2.), (-1.5, 1.5), 80, True)
    contour_grid((-.8, .6), (-.5, 0.), .06+.003*jnp.arange(21), False)

    start = time.perf_counter()
    result = fminsearch(lambda v: minfun(v[0], v[1]), [.41, -.2],
                        tol_x=1e-14, tol_fun=1e-4)
    print(f'Elapsed time is {time.perf_counter()-start:.6f} seconds.', flush=True)
    xvec, yval = result.x, result.fun
    print('yval =', flush=True)
    print(f'   {float(yval):.15f}', flush=True)
    fig, ax = plt.subplots(figsize=(6, 2.7))
    plotneedle(ax, float(xvec[0]), float(xvec[1]))
    ax.axis([-2, 2, -.4, 1.2])
    ax.plot(float(xvec[0]), float(yval), '.k', markersize=5)
    ax.set_yticks([-.4, -.2, 0., .2, .4, .6, .8, 1., 1.2])
    ax.grid(True)
    save(fig)


if __name__ == '__main__':
    run()
