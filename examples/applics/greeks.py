"""Accurate Greeks, translated from applics/Greeks.m (Pachon, 2014).

Original: https://www.chebfun.org/examples/applics/Greeks.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import os
import sys

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np  # uses-numpy: projected Matplotlib mesh vertices and colors only.
from jax.scipy.special import ndtr
from matplotlib.projections import register_projection
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_IMG = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'images', 'applics')
POSITIONS = ([.085, .56, .40, .39], [.57, .56, .40, .39],
             [.085, .08, .40, .39], [.57, .08, .40, .39])
ST, VOL, TAU, MU, R, K = 100, .45, .5, .07, .01, 100
LEVELS = (ST, VOL, TAU, MU)


def f_pdf(s, st, vol, tau, mu):
    # Keep the source's removable 0/0 at s=0 for Chebfun to resolve.
    return (jnp.exp(-(jnp.log(s / st) - (mu - .5 * vol**2) * tau)**2
                    / (2 * vol**2 * tau))
            / (vol * s * jnp.sqrt(2 * jnp.pi * tau)))


def _save(fig, number):
    _historical_render(fig, number)
    _savefig(fig, os.path.join(_IMG, f'Greeks_{number:02d}.png'), size=(600, 268))
    plt.close(fig)


class _InterpolatedSurface(Poly3DCollection):
    """MATLAB facecolor=interp on the existing mesh, using Gouraud triangles."""

    def draw(self, renderer):
        if not self.get_visible():
            return
        x, y, z = proj3d.proj_transform(*self._vec[:3], self.axes.M)
        ids = np.asarray([[section.start, section.start + k, section.start + k + 1]
                          for section in self._segslices
                          for k in range(1, section.stop - section.start - 1)])
        if not ids.size:
            return
        triangles = np.column_stack((x, y))[ids]
        colors = self.cmap(self.norm(self._vec[2]))[ids]
        # Match the existing surface painter's depth order; interpolate colors
        # between the original vertex values, with no function reevaluation.
        order = np.argsort(np.mean(z[ids], axis=1))[::-1]
        gc = renderer.new_gc()
        self._set_gc_clip(gc)
        renderer.draw_gouraud_triangles(gc, triangles[order], colors[order],
                                       self.get_transform())
        gc.restore()
        self.stale = False


def _clip_surface_x(collection, lo, hi):
    """Clip existing mesh faces geometrically; never resample the function."""
    vertices = collection._vec[:3]
    polygons, kept = [], []
    for i, section in enumerate(collection._segslices):
        polygon = vertices[:, section].T.tolist()
        for bound, sign in ((lo, 1), (hi, -1)):
            clipped = []
            if polygon:
                previous = polygon[-1]
                prev_inside = sign * (previous[0] - bound) >= 0
                for point in polygon:
                    inside = sign * (point[0] - bound) >= 0
                    if inside != prev_inside:
                        fraction = (bound - previous[0]) / (point[0] - previous[0])
                        clipped.append([bound] + [previous[k] + fraction * (point[k] - previous[k]) for k in (1, 2)])
                    if inside:
                        clipped.append(point)
                    previous, prev_inside = point, inside
            polygon = clipped
        if len(polygon) >= 3:
            polygons.append(polygon)
            kept.append(i)
    # Scalar colors retain the original full-domain mesh's face values and norm.
    values = collection.get_array()
    collection.set_verts(polygons)
    if values is not None:
        collection.set_array(values[kept])


def _historical_render(fig, number):
    # Visible limits/ticks transcribed from the five published 600x268 PNGs.
    # This only adapts plotting autoscale differences; domains stay literal.
    if number in (1, 3, 5):
        ticks = ([[-4e-4, -3e-4, -2e-4, -1e-4, 0, 1e-4, 2e-4],
                  [-.03, -.02, -.01, 0, .01, .02, .03],
                  [-.015, -.01, -.005, 0, .005, .01, .015],
                  [-.02, -.015, -.01, -.005, 0, .005, .01]] if number != 5 else
                 [[0, .5e-4, 1e-4, 1.5e-4, 2e-4],
                  [-.03, -.025, -.02, -.015, -.01, -.005, 0, .005, .01],
                  [-.005, 0, .005, .01, .015],
                  [-.004, -.002, 0, .002, .004, .006, .008]])
        for ax, yticks in zip(fig.axes, ticks):
            ax.set_xticks(range(0, 201, 20))
            ax.set_xlim(.01 if number == 3 else 0, 200)
            ax.set_yticks(yticks)
            ax.set_ylim(yticks[0], yticks[-1])
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis='y', scilimits=(-3, -3) if number == 5 and ax in fig.axes[2:] else (-3, 3))
    else:
        bounds = ([(90, 110), (.4, .5), (.45, .55), (.06, .08)] if number == 2 else
                  [(80, 120), (.41, .5), (.49, .51), (.006, .013)])
        yticks = ([[90, 95, 100, 105, 110], [.4, .45, .5], [.45, .5, .55], [.06, .065, .07, .075, .08]] if number == 2 else
                  [[80, 90, 100, 110, 120], [.42, .44, .46, .48, .5], [.49, .495, .5, .505, .51], [.006, .008, .01, .012]])
        zticks = ([[-5e-4, 0, 5e-4], [-.04, -.02, 0, .02, .04], [-.02, -.01, 0, .01, .02], [-.02, -.01, 0, .01]] if number == 2 else
                  [[-2e-4, 0, 2e-4, 4e-4], [-.04, -.02, 0, .02], [-.005, 0, .005, .01, .015], [-.005, 0, .005, .01]])
        for ax, domain, yy, zz in zip(fig.axes, bounds, yticks, zticks):
            ax.set_xticks([0, 50, 100, 150, 200])
            ax.set_xlim(.01 if number == 2 else 0, 200)
            ax.set_yticks(yy)
            ax.set_ylim(*domain)
            ax.set_zticks(zz)
            ax.set_zlim(zz[0], zz[-1])
            ax.zaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis='z', scilimits=(-3, 3))
            scale = (1e-4 if ax is fig.axes[0] else
                     1e-3 if number == 4 and ax in fig.axes[2:] else None)
            if scale is not None:
                ax.set_zticklabels([f'{value / scale:g}' for value in zz])
                exponent = -4 if scale == 1e-4 else -3
                ax.text2D(-.055, .96, rf'$\times10^{{{exponent}}}$',
                          transform=ax.transAxes, fontsize=3.6)
            for collection in ax.collections:
                # Source CLim comes from full-domain vertex data, before xlim.
                collection.set_clim(float(collection._vec[2].min()),
                                    float(collection._vec[2].max()))
                collection.__class__ = _InterpolatedSurface
                if number == 4:
                    _clip_surface_x(collection, 0, 200)


class _SourceAxes3D(Axes3D):
    """Keep MATLAB's explicit rectangular Position during 3D drawing."""

    name = 'greeks_source_3d'

    def apply_aspect(self, position=None):
        # Axes3D otherwise shrinks every rectangle to a square at draw time.
        # MATLAB stretches the projected scene over its specified Position.
        if position is None:
            position = self.get_position(original=True)
        self._set_position(position, which='active')


register_projection(_SourceAxes3D)


def _axes(fig, position, surface=False):
    ax = fig.add_axes(position, projection='greeks_source_3d' if surface else None)
    # Historical publication scales source 14-point text to about five pixels.
    # Keep source normalized positions; this is a rendering adapter only.
    ax.tick_params(labelsize=3.6, length=1.4, width=.3, pad=1)
    ax.set_xlabel(r'$S_T$', fontsize=3.6, labelpad=0)
    if not surface:
        ax.tick_params(top=True, right=True, direction='in')
        for spine in ax.spines.values():
            spine.set_color('.5')
            spine.set_linewidth(.3)
        ax.ticklabel_format(axis='y', scilimits=(-3, 3))
        ax.yaxis.get_offset_text().set_fontsize(3.6)
    return ax


def _lines(functions, labels, coords, number, fontsize=14, xlim=None):
    fig = plt.figure(figsize=(6, 2.68))
    for fn, label, coord, pos in zip(functions, labels, coords, POSITIONS):
        ax = _axes(fig, pos)
        matlab_plot(fn, ax=ax, linewidth=1.6 * .36)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.text(*coord, label, fontsize=fontsize * .36)
    _save(fig, number)


def _surfaces(parts, labels, coords, ylabels, number, xlim=None):
    fig = plt.figure(figsize=(6, 2.68))
    for fn, label, coord, ylabel, pos in zip(parts, labels, coords, ylabels, POSITIONS):
        ax = _axes(fig, pos, surface=True)
        # @separableApprox/surf.m uses 200 uniform points on the FULL domain.
        fn.plot(ax=ax, n_pts=200)
        ax.set_position(pos)
        ax.set_xlabel(r'$S_T$', fontsize=3.6, labelpad=0)
        ax.set_ylabel(ylabel, fontsize=3.6, labelpad=0)
        ax.tick_params(labelsize=3.6, pad=0)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.text(*coord, label, fontsize=24 * .36)
    for ax, pos in zip(fig.axes, POSITIONS):
        ax.set_position(pos)
    _save(fig, number)


def _payoff_parts(w, domx):
    # minS=0 is assigned by the call cell and retained in the put cell.
    slices = [
        Chebfun2.from_function(lambda s, st: jnp.exp(-R*TAU)*f_pdf(w*s+K, st, VOL, TAU, R), domain=(0, domx, 80, 120)),
        Chebfun2.from_function(lambda s, v: jnp.exp(-R*TAU)*f_pdf(w*s+K, ST, v, TAU, R), domain=(0, domx, .41, .5)),
        Chebfun2.from_function(lambda s, t: -jnp.exp(-R*t)*f_pdf(w*s+K, ST, VOL, t, R), domain=(0, domx, .49, .51)),
        Chebfun2.from_function(lambda s, r: jnp.exp(-r*TAU)*f_pdf(w*s+K, ST, VOL, TAU, r), domain=(0, domx, .006, .013)),
    ]
    parts = [f.diff(dim=1) for f in slices]
    curves = [p(':', level).transpose() for p, level in zip(parts, (ST, VOL, TAU, R))]
    xx = chebfun('x', domain=(0, domx))
    return [float((xx*g).sum()) for g in curves], parts, curves


def _exact(w):
    d1 = (jnp.log(ST/K)+(R+.5*VOL**2)*TAU)/(VOL*jnp.sqrt(TAU))
    d2 = d1-VOL*jnp.sqrt(TAU)
    pdf1 = jnp.exp(-d1*d1/2)/jnp.sqrt(2*jnp.pi)
    pdf2 = jnp.exp(-d2*d2/2)/jnp.sqrt(2*jnp.pi)
    return [float(w*ndtr(w*d1)),
            float(K*jnp.exp(-R*TAU)*pdf2*jnp.sqrt(TAU)),
            float(-ST*pdf1*VOL/(2*jnp.sqrt(TAU))-w*R*K*jnp.exp(-R*TAU)*ndtr(w*d2)),
            float(w*K*TAU*jnp.exp(-R*TAU)*ndtr(w*d2))]


def run():
    os.makedirs(_IMG, exist_ok=True)
    pdf = chebfun(lambda s: f_pdf(s, ST, VOL, TAU, MU), domain=(0, 200))
    steps = (.001, .0001, .0001, .0001)
    bumped = []
    for i, step in enumerate(steps):
        params = list(LEVELS)
        params[i] += step
        bumped.append((chebfun(lambda s, p=params: f_pdf(s, *p), domain=(0, 200))-pdf)/step)
    _lines(bumped, [r'$[f(S+\delta S)-f(S)]/\delta S$',
                    r'$[f(\sigma+\delta\sigma)-f(\sigma)]/\delta\sigma$',
                    r'$[f(t+\delta t)-f(t)]/\delta t$',
                    r'$[f(\mu+\delta\mu)-f(\mu)]/\delta\mu$'],
           [(100, -.0003), (100, .015), (100, .01), (100, -.015)], 1)
    bounds = [(90, 110), (.4, .5), (.45, .55), (.06, .08)]
    slices = []
    for i, bound in enumerate(bounds):
        def fn(s, v, i=i):
            params = list(LEVELS)
            params[i] = v
            return f_pdf(s, *params)
        slices.append(Chebfun2.from_function(fn, domain=(.01, 200, *bound)))
    partials = [f.diff(dim=1) for f in slices]
    _surfaces(partials, [r'$\partial f/\partial S$', r'$\partial f/\partial\sigma$',
                         r'$\partial f/\partial\tau$', r'$\partial f/\partial\mu$'],
              [(110, 92, -.0003), (10, .42, -.03), (-50, .44, -.006), (100, .06, -.01)],
              [r'$S_t$', r'$\sigma$', r'$\tau$', r'$\mu$'], 2)
    _lines([p(':', level) for p, level in zip(partials, LEVELS)],
           [r'$[\partial f/\partial S]_{S=100}$', r'$[\partial f/\sigma\mu]_{\sigma=0.45}$',
            r'$[\partial f/\partial\tau]_{\tau=0.5}$', r'$[\partial f/\partial\mu]_{\mu=0.07}$'],
           [(100, -.0003), (100, .015), (100, .01), (100, -.015)], 3)
    call, parts, curves = _payoff_parts(1, 5000)
    _surfaces(parts, [r'$\partial g/\partial S$', r'$\partial g/\partial\sigma$',
                      r'$\partial f/\partial\tau$', r'$\partial f/\partial r$'],
              [(80, 112, -.0003), (100, .42, -.03), (100, .49, .015), (100, .005, .009)],
              [r'$S_t$', r'$\sigma$', r'$\tau$', '$r$'], 4, xlim=(0, 200))
    _lines(curves, [r'$[\partial g/\partial S]_{S=100}$', r'$[\partial g/\sigma\mu]_{\sigma=0.45}$',
                    r'$[\partial g/\partial\tau]_{\tau=0.5}$', r'$[\partial g/\partial r]_{r=0.01}$'],
           [(100, .0001), (100, -.015), (100, .01), (100, .005)], 5, fontsize=18, xlim=(0, 200))
    for label, val in zip(('delta approx ', 'vega approx  ', 'theta approx ', 'rho approx   '), call):
        print(f'{label}[call] = {val:.15f}')
    put, _, _ = _payoff_parts(-1, 99)
    for label, val in zip(('delta approx ', 'vega approx  ', 'theta approx ', 'rho approx   '), put):
        print(f'{label}[put] = {val:.15f}')
    exact_call, exact_put = _exact(1), _exact(-1)
    print('                       call                 put')
    for name, gap, ec, ac, ep, ap in zip(('delta', 'vega', 'theta', 'rho'),
                                        (5, 4, 3, 4), exact_call, call, exact_put, put):
        print(f'{name + " exact":<13}: {ec:.15f}'+' '*gap+f'{ep:.15f}')
        print(f'{name + " approx":<13}: {ac:.15f}'+' '*gap+f'{ap:.15f}')
        print(f'{name + " error":<13}: {abs((ec-ac)/ec):.5g}            {abs((ep-ap)/ep):.5g}')
        print('-'*55)
    return {'call': call, 'put': put, 'exact_call': exact_call, 'exact_put': exact_put}


if __name__ == '__main__':
    run()
