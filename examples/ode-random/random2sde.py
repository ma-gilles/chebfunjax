"""From random functions to SDEs.

Translation of ode-random/Random2SDE.m by Nick Trefethen and
Abdul-Lateef Haji-Ali (May 2017): three "smooth random walk" sample
paths -- cumsum of normalized ('big') random functions with
lambda = 0.001 on [0, 1] -- which for small lambda look to the eye
like Brownian motion, the Stratonovich SDE limit.

rng(0) seeds numpy's MT19937 identically, but MATLAB's randn
normal transform differs from numpy's, so the paths differ.

Original: https://www.chebfun.org/examples/ode-random/Random2SDE.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun1d.randfuns import randnfun
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-random')

# ``help randnfun``: the help text of Chebfun's randnfun.m.
HELP_RANDNFUN = """\
 RANDNFUN   Smooth random function
    F = RANDNFUN(LAMBDA) returns a CHEBFUN on [-1,1] with maximum
    frequency <= 2pi/LAMBDA and standard normal distribution N(0,1)
    at each point.  F can be regarded as a sample path of a Gaussian
    process.  It is obtained by calling RANDNFUN(LAMBDA, 'trig') on an
    interval 20% longer and restricting the result to [-1,1].

    RANDNFUN(LAMBDA, DOM) returns a result with domain DOM = [A, B].

    RANDNFUN(LAMBDA, N) returns a quasimatrix with N independent columns.

    RANDNFUN(LAMBDA, 'big') normalizes the output by dividing it by
    SQRT(LAMBDA/2), so white noise is approached in the limit LAMBDA -> 0,
    with an indefinite integral corresponding to standard Brownian motion.

    RANDNFUN(LAMBDA, 'trig') returns a random periodic function.  This
    is defined by a finite Fourier-Wiener series with independent normally
    distributed coefficients of equal variance.

    RANDNFUN(LAMBDA, 'complex') returns a complex random function.  The
    variance is the same as in the real case (i.e., not twice as great).

    RANDNFUN() uses the default value LAMBDA = 1.  Combinations such
    as RANDNFUN(DOM) and RANDNFUN('big', LAMBDA) are allowed so long as
    N, if present, is preceded by an explicit specification of LAMBDA.

    Reference: S. Filip, A. Javeed, and L. N. Trefethen, "Smooth random
    functions, random ODEs, and Gaussian processes," SIAM Review, 61
    (2019), pp. 185-205.

  Examples:

    f = randnfun(0.1); std(f), plot(f)
    plotcoeffs(f, '.'), xlim([0 200])

    X = randnfun(.01,2); cov(X)

    s = randi(100);
    rng(s), f1 = randnfun(0.5,'big',[0 10],3);
    rng(s), f2 = randnfun(0.1,'big',[0 10],3);
    plot(cumsum(f1),'k',cumsum(f2),'r')

    plot(cumsum(randnfun(.01,[0 5],'complex','big'))), axis equal

  See also RANDNFUN2, RANDNFUNSPHERE, RANDNFUNDISK, SMOOTHIE.
"""


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    print(HELP_RANDNFUN)

    t0 = time.perf_counter()                       # tic
    np.random.seed(5489)                           # rng(0)
    u = randnfun(0.001, [0, 1], 3, 'big')
    fig, ax = plt.subplots()
    matlab_plot(u.cumsum(), ax=ax)
    ax.grid(True)
    ax.set_ylim(-2, 2)
    ax.set_xlabel("t")
    ax.set_ylabel("u")
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "Random2SDE_01.png"), size=(600, 270))
    plt.close(fig)

    total_time_in_seconds = time.perf_counter() - t0   # toc
    print("total_time_in_seconds =")
    print(f"{total_time_in_seconds:20.15f}")


if __name__ == "__main__":
    run()
