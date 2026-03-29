"""Generate plots for Guide Chapter 20: Ballfun.

Uses the library's plot_ball_slices() which faithfully translates
MATLAB @ballfun/plot.m with correct elevation coordinates.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import jax.numpy as jnp
from chebfunjax.ballfun import Ballfun
from chebfunjax.plotting import plot_ball_slices, chebfun_style
chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT, exist_ok=True)
plot_num = 0


def save(fig, desc):
    global plot_num
    plot_num += 1
    fname = os.path.join(OUT, f'guide20_{plot_num:02d}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved {fname}: {desc}")


funcs = [
    (lambda x, y, z: x**2 + y**2 + z**2, r"$x^2+y^2+z^2$", "r^2"),
    (lambda x, y, z: jnp.cos(jnp.pi*x) * jnp.sin(jnp.pi*y) * jnp.exp(z),
     r"$\cos(\pi x)\sin(\pi y)e^z$", "cos*sin*exp"),
    (lambda x, y, z: 3*z**2 - (x**2 + y**2 + z**2), r"$Y_2^0$", "solid harmonic"),
    (lambda x, y, z: jnp.exp(-5*(x**2 + y**2 + z**2)), r"$e^{-5r^2}$", "Gaussian"),
    (lambda x, y, z: x * y * z, r"$xyz$", "xyz"),
    (lambda x, y, z: jnp.sinc(3*jnp.sqrt(x**2+y**2+z**2)/jnp.pi),
     r"$\mathrm{sinc}(3r)$", "sinc"),
]

for fn, title, desc in funcs:
    try:
        bf = Ballfun.from_function(fn, fixed_size=(15, 16, 16))
        fig, ax = plot_ball_slices(bf, title=title)
        save(fig, desc)
    except Exception:
        traceback.print_exc()
        plot_num += 1
        print(f"  SKIP plot {plot_num}")

print(f"\nGuide 20: Generated {plot_num} plots.")
