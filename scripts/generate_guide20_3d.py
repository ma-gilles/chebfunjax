"""Generate MATLAB-style 3D slice plots for Guide 20 (Ballfun).

Produces the characteristic ball-slice visualization matching MATLAB's
Ballfun plot: slices at constant r, theta, lambda displayed in 3D
with sph2cart conversion, LightSource shading (camlight + phong), and
equal-aspect axes.

Uses the faithful MATLAB translation in plot_ball_slices().
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from chebfunjax.plotting import chebfun_style, plot_ball_slices
chebfun_style()

import jax.numpy as jnp
from chebfunjax.ballfun.ballfun import Ballfun

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT, exist_ok=True)


def save(fig, fname, desc):
    path = os.path.join(OUT, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}: {desc}")


# Plot 1: f = x^2 + y^2 + z^2 (radial function)
try:
    f1 = Ballfun.from_function(lambda x, y, z: x**2 + y**2 + z**2)
    fig, ax = plot_ball_slices(f1, title=r"$f(x,y,z) = x^2+y^2+z^2$", n_pts=60)
    save(fig, "guide20_01.png", "r^2 ball slices")
except Exception as e:
    print(f"  SKIP plot 1: {e}")

# Plot 2: f = cos(pi*x)*sin(pi*y)*exp(z)
try:
    f2 = Ballfun.from_function(
        lambda x, y, z: jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.exp(z))
    fig, ax = plot_ball_slices(f2, title=r"$\cos(\pi x)\sin(\pi y)e^z$", n_pts=60)
    save(fig, "guide20_02.png", "cos*sin*exp ball slices")
except Exception as e:
    print(f"  SKIP plot 2: {e}")

# Plot 3: Solid harmonic Y_2^0 = (3z^2 - r^2)
try:
    f3 = Ballfun.from_function(
        lambda x, y, z: 3 * z**2 - (x**2 + y**2 + z**2))
    fig, ax = plot_ball_slices(f3, title=r"$Y_2^0: 3z^2 - r^2$", n_pts=60)
    save(fig, "guide20_03.png", "solid harmonic ball slices")
except Exception as e:
    print(f"  SKIP plot 3: {e}")

# Plot 4: Gaussian exp(-5r^2)
try:
    f4 = Ballfun.from_function(
        lambda x, y, z: jnp.exp(-5 * (x**2 + y**2 + z**2)))
    fig, ax = plot_ball_slices(f4, title=r"$e^{-5r^2}$", n_pts=60)
    save(fig, "guide20_04.png", "Gaussian ball slices")
except Exception as e:
    print(f"  SKIP plot 4: {e}")

# Plot 5: x*y*z (mixed)
try:
    f5 = Ballfun.from_function(lambda x, y, z: x * y * z)
    fig, ax = plot_ball_slices(f5, title=r"$f = xyz$", n_pts=60)
    save(fig, "guide20_05.png", "xyz ball slices")
except Exception as e:
    print(f"  SKIP plot 5: {e}")

# Plot 6: sinc(3r)
try:
    f6 = Ballfun.from_function(
        lambda x, y, z: jnp.sinc(3 * jnp.sqrt(x**2 + y**2 + z**2) / jnp.pi))
    fig, ax = plot_ball_slices(f6, title=r"$\mathrm{sinc}(3r)$", n_pts=60)
    save(fig, "guide20_06.png", "sinc ball slices")
except Exception as e:
    print(f"  SKIP plot 6: {e}")

print(f"\nGuide 20 (3D): Generated 6 plots.")
