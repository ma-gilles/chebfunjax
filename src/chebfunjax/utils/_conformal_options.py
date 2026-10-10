"""Optional plotting and diagnostics from conformal.m, Chebfun7574c77.

Copyright 2019 by The University of Oxford and The Chebfun Developers.
The Python keyword interface preserves the native plots/numbers branches.
"""
from __future__ import annotations

import jax.numpy as jnp


def plot_conformal(C, ctr, scl, finv, pol, polinv):
    """Clear current figure and draw native pole/grid geometry, lines141–184."""
    import matplotlib.pyplot as plt
    from matplotlib.markers import MarkerStyle

    from chebfunjax.plotting import _matlab_ticks, matlab_plot
    from chebfunjax.utils.quadrature import chebpts

    fig = plt.gcf()
    fig.clear()
    circ = jnp.exp(2j * jnp.pi * jnp.arange(301) / 300)
    left = fig.add_axes([.04, .38, .45, .53])
    matlab_plot(C, 'b', ax=left, linewidth=1)
    radius = 1.4 * float(jnp.real(scl))
    left.set_xlim(float(jnp.real(ctr))-radius, float(jnp.real(ctr))+radius)
    left.set_ylim(float(jnp.imag(ctr))-radius, float(jnp.imag(ctr))+radius)
    left.set_box_aspect(1)
    # Native Line point diameter is MarkerSize/3; see matlab_spy provenance.
    point = MarkerStyle('.').scaled(2/3)
    left.plot(jnp.real(pol), jnp.imag(pol), color='r', linestyle='none',
              marker=point, markersize=8, markeredgewidth=0)
    left.set_title(f'{len(pol)} poles', fontweight='normal')
    _matlab_ticks(left)

    right = fig.add_axes([.52, .38, .45, .53])
    right.plot(circ.real, circ.imag, 'b', linewidth=1)
    right.set_xlim(-1.6, 1.6)
    right.set_ylim(-1.6, 1.6)
    right.set_box_aspect(1)
    _matlab_ticks(right)
    right.set_xticks([-1, 0, 1])
    right.set_yticks([-1, 0, 1])
    right.plot(jnp.real(polinv), jnp.imag(polinv), color='r', linestyle='none',
               marker=point, markersize=8, markeredgewidth=0)
    right.set_title(f'{len(polinv)} poles', fontweight='normal')

    for k in range(1, 8):
        r = k / 8
        image = finv(r * circ)
        left.plot(jnp.real(image), jnp.imag(image), '-k', linewidth=.5)
        right.plot(jnp.real(r * circ), jnp.imag(r * circ), '-k', linewidth=.5)
    ray = chebpts(301)
    ray = ray[ray >= 0]
    for k in range(1, 17):
        points = ray * jnp.exp(2j * jnp.pi * k / 16)
        image = finv(points)
        left.plot(jnp.real(image), jnp.imag(image), '-k', linewidth=.5)
        right.plot(jnp.real(points), jnp.imag(points), '-k', linewidth=.5)
    # Source axes(h1), hold off; axes(h2), hold off ends on the right axes.
    plt.sca(right)
    return fig, (left, right)


def print_conformal_numbers(f, finv, Z, W, pol, polinv, tcomp, tplot, M, err, poly):
    """Print source quantities in native order/format, conformal.m190–216."""
    print(' ')
    print(f'                         computation time in seconds:{tcomp:6.2f}')
    if tplot is not None:
        print(f'                            plotting time in seconds:{tplot:6.2f}')
    print(f'            number of sample points Z on boundary, M:  {M:d}')
    print(f'                      numbers of poles of f and finv:  {len(pol):d}, {len(polinv):d}')
    error = float(jnp.linalg.norm(Z - finv(f(Z)), ord=jnp.inf))
    print(f'back-and-forth boundary error norm(Z-finv(f(Z)),inf):  {error:6.1e}')
    error = float(jnp.linalg.norm(W - f(finv(W)), ord=jnp.inf))
    print(f' inverse back-and-forth error norm(W-f(finv(W)),inf):  {error:6.1e}')
    error = float(jnp.linalg.norm(.9 * W - f(finv(.9 * W)), ord=jnp.inf))
    print(f' interior inverse error norm(.9*W-f(finv(.9*W)),inf):  {error:6.1e}')
    if poly is not None:
        print(f'             max error of least-squares problem, err:  {err:6.1e}')
        n = poly['n']
        print(f'                                polynomial degree, n:  {n:d}')
        print(f'                number of real degrees of freedom, N:  {2*n+1:d}')
        condition = float(jnp.linalg.cond(poly['A']))
        print(f'          condition number of least-squares matrix A:  {condition:6.1e}')
    else:
        print(f'                            rough error measure, err:  {err:6.1e}')
    print(' ')
