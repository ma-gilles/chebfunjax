"""Vandermonde with Arnoldi: source translation, full degree80.

Pablo Brubeck, Yuji Nakatsukasa, and Nick Trefethen, January2020.
Chebfun7574c77680d7e82b79626300bf255498271a72df;
example f4b9ea46cfc2f52f20a844627f4a74d0bb10098c.
Original: https://www.chebfun.org/examples/linalg/VandermondeArnoldi.html
Copyright The University of Oxford and The Chebfun Developers.
"""
import os
import sys

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import mldivide
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.quadrature import chebpts

_IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                    "docs", "images", "linalg")


def _scalar(value):
    # MATLAB one-element numeric assignment versus JAX rank-zero storage.
    value = jnp.asarray(value)
    if value.size != 1:
        raise ValueError("Expected a scalar numeric result")
    return value.reshape(())


def _ans(value, scientific=True, name="ans"):
    print(f"{name} =")
    print(f"   {float(_scalar(value)):.4e}" if scientific
          else f"    {float(_scalar(value)):.4f}")


def polyfit(x, f, n):
    A = x ** jnp.arange(n+1)
    return mldivide(A, f)


def polyval(c, s):
    n = len(c)-1
    B = s ** jnp.arange(n+1)
    return B @ c


def polyfitA(x, f, n):
    Q = 1 + 0*x
    H = jnp.zeros((n+1, n))
    for k in range(n):
        q = x * Q.extract_columns(k)
        for j in range(k+1):
            H = H.at[j, k].set(_scalar(Q.extract_columns(j).H @ q))
            q = q - H[j, k] * Q.extract_columns(j)
        H = H.at[k+1, k].set(_scalar(q.norm()))
        Q = cj.Chebfun.horzcat([Q, q/H[k+1, k]])
    return mldivide(Q, f), H


def polyvalA(d, H, s):
    W = 1 + 0*s
    for k in range(H.shape[1]):
        w = s * W.extract_columns(k)
        for j in range(k+1):
            w = w - H[j, k] * W.extract_columns(j)
        W = cj.Chebfun.horzcat([W, w/H[k+1, k]])
    return W @ d


def run():
    chebfun_style()
    os.makedirs(_IMG, exist_ok=True)
    for count in (17, 33):
        _ans(jnp.linalg.cond(jnp.vander(chebpts(count))))
    for count in (17, 33):
        _ans(jnp.linalg.cond(chebpts(count)[:, None] ** jnp.arange(count)))
    for count in (17, 33):
        _, constant = cj.lebesgue(chebpts(count), return_constant=True)
        _ans(constant, scientific=False, name=f"L{count-1}")
    x = cj.chebfun("x")
    for count in (17, 33):
        _ans(x.vander(count).cond())
    for count in (17, 33):
        _ans((x ** jnp.arange(count)).cond())
    for count in (17, 33):
        _ans(jnp.linalg.cond(jnp.vander(jnp.linspace(-1., 1., count))))
    f = abs(x)
    c = polyfit(x, f, 80)
    y = polyval(c, x)
    print("y =")
    print(repr(y))
    _ans(y.max()[1], scientific=False)
    _ans(jnp.linalg.norm(c, ord=jnp.inf))
    d, H = polyfitA(x, f, 80)
    yA = polyvalA(d, H, x)
    print("yA =")
    print(repr(yA))
    fig, ax = plt.subplots()
    matlab_plot(cj.Chebfun.horzcat([y, yA]), ax=ax)
    _savefig(fig, os.path.join(_IMG, "VandermondeArnoldi_01.png"), size=(600, 253))
    plt.close(fig)


if __name__ == "__main__":
    run()
