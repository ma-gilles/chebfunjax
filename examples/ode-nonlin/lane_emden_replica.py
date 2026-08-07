"""The Lane-Emden equation from astrophysics (partial replica).

n = 0 and n = 1 reproduce their closed forms to near machine precision;
n >= 2 is blocked on the ledgered singular-endpoint Jacobian defect --
see docs/examples/ode-nonlin/LaneEmden.md for the full statement.

Original: https://www.chebfun.org/examples/ode-nonlin/LaneEmden.html
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
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
warnings.filterwarnings("ignore")
IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docs', 'images', 'ode-nonlin')
os.makedirs(IMG, exist_ok=True)
xx = np.linspace(0, 10, 2000)
fig, ax = plt.subplots(figsize=(8.6, 4.6))
for n in (0, 1):
    N = Chebop(lambda x, u, _n=n: x*u.diff(2) + 2*u.diff() + x*u**_n,
               domain=(0.0, 10.0))
    N.lbc = lambda u: [u - 1, u.diff()]
    u = N.solvebvp(0.0)[0]
    ax.plot(xx, np.asarray(u(xx)), lw=2, label=f"n={n}")
exact = {2: None}
ax.plot(xx, 1/np.sqrt(1 + xx**2/3), "--", lw=1.5,
        label="n=5 exact (not reached)")
ax.axis([0, 10, -1, 1])
ax.grid(True)
ax.set_title("Solution of the Lane-Emden equation")
ax.set_xlabel("x"); ax.set_ylabel("u")
ax.legend()
fig.set_facecolor("white"); fig.tight_layout()
fig.savefig(os.path.join(IMG, "LaneEmden_repl_01.png"), dpi=150,
            bbox_inches="tight")
print("figure saved", flush=True)
for n in (0, 1):
    N = Chebop(lambda x, u, _n=n: x*u.diff(2) + 2*u.diff() + x*u**_n,
               domain=(0.0, 10.0))
    N.lbc = lambda u: [u - 1, u.diff()]
    u = N.solvebvp(0.0)[0]
    ex = (1 - xx**2/6) if n == 0 else np.where(xx == 0, 1, np.sin(xx)/np.maximum(xx, 1e-300))
    print(f"n={n}: len={len(u)} maxerr={float(np.max(np.abs(np.asarray(u(xx)) - ex))):.2e}")
print("DONE", flush=True)
