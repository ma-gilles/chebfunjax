# chebfun_smoke.py — convert to a Jupyter notebook with:
#   jupytext --to notebook chebfun_smoke.py
# or run directly:
#   python chebfun_smoke.py
#
# Tests the top-level plotting API exposed by chebfunjax.

# %%
import matplotlib
matplotlib.use("Agg")  # headless on HPC — override here for script use only

import chebfunjax as cj
import jax.numpy as jnp

# %%
# 1-D Chebfun: sin(x)
f = cj.chebfun(jnp.sin)
fig, ax = cj.plot(f, title="sin(x)")
fig.savefig("smoke_plot_sin.png", dpi=72)
print("plot(sin): OK")

# %%
# Chebyshev coefficient decay
fig, ax = cj.plotcoeffs(f)
fig.savefig("smoke_plotcoeffs_sin.png", dpi=72)
print("plotcoeffs(sin): OK")

# %%
# 2-D Chebfun: cos(x + y)
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
g = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
fig, ax = cj.surf(g, title="cos(x+y)")
fig.savefig("smoke_surf_cos.png", dpi=72)
print("surf(cos(x+y)): OK")

# %%
fig, ax = cj.contour(g, title="cos(x+y) contours")
fig.savefig("smoke_contour_cos.png", dpi=72)
print("contour(cos(x+y)): OK")

# %%
# Phase plot of z^2
fig, ax = cj.phaseplot(lambda z: z**2, region=[-2.0, 2.0, -2.0, 2.0],
                       title="phase of z^2")
fig.savefig("smoke_phaseplot_z2.png", dpi=72)
print("phaseplot(z^2): OK")

# %%
print("All smoke tests passed.")
