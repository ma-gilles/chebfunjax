"""Conformal map to a square.

Translation of complex/ConformalSquare.m by Toby Driscoll (January 2013).
The literal Schwarz-Christoffel derivative uses public Chebfun power and
integration. Representation warnings are emitted normally.

Original: https://www.chebfun.org/examples/complex/ConformalSquare.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import _source_plot_line_and_points, chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'complex')
ZPRE = np.array([1, 1j, -1, -1j])
FIG = [0]
DERIVATIVE = [0]

def _capture(cf, name):
    destination = os.environ.get("CHEBFUNJAX_CAPTURE_DIR")
    if not destination:
        return cf
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    x = np.linspace(float(cf.domain.breakpoints[0]), float(cf.domain.breakpoints[-1]), 501)
    arrays = {"sample_x": x, "sample_values": np.asarray(cf(x)),
              "point_values": np.asarray(cf.point_values)}
    pieces = []
    for i, piece in enumerate(cf.funs):
        tech = piece.tech
        smooth = getattr(tech, 'smoothPart', tech)
        arrays[f"piece_{i}_coeffs"] = np.asarray(smooth.coeffs)
        arrays[f"piece_{i}_values"] = np.asarray(smooth.values)
        pieces.append({"interval": list(piece.interval), "tech": type(tech).__name__,
                       "length": len(tech), "exponents": list(getattr(tech, 'exponents', ()))})
    np.savez(root / (name + '.npz'), **arrays)
    (root / (name + '.json')).write_text(json.dumps({"domain": list(cf.domain.breakpoints),
        "length": len(cf), "representation": repr(cf), "isreal": bool(cf.isreal()), "pieces": pieces}, indent=2))
    print("@@CONFORMAL_STAGE:" + name + "@@", flush=True)
    return cf

def fprime(z):
    DERIVATIVE[0] += 1
    name = f"{DERIVATIVE[0]:02d}"
    _capture(z, 'input_' + name)
    # Exact original four-divisor expression; multiplication is elementwise
    # here because each Chebfun is a scalar column.
    derivative = ((1 - z / ZPRE[0]) * (1 - z / ZPRE[1])
                  * (1 - z / ZPRE[2]) * (1 - z / ZPRE[3])) ** (-0.5)
    return _capture(derivative, 'derivative_' + name)

def _snapshot(fig):
    FIG[0] += 1
    fig.canvas.draw()
    destination = os.environ.get("CHEBFUNJAX_CAPTURE_DIR")
    if destination:
        arrays = {}
        metadata = []
        for ai, ax in enumerate(fig.axes):
            metadata.append({"position": list(ax.get_position().bounds), "limits": list(ax.axis()),
                             "xticks": ax.get_xticks().tolist(), "yticks": ax.get_yticks().tolist()})
            for li, line in enumerate(ax.lines):
                arrays[f"axes_{ai}_line_{li}"] = line.get_xydata()
                metadata[-1].setdefault('lines', []).append({"color": line.get_color(),
                    "line_width": line.get_linewidth(), "marker_size": line.get_markersize(),
                    "marker_edge_width": line.get_markeredgewidth(), "style": line.get_linestyle()})
        root = Path(destination)
        np.savez(root / f"figure_{FIG[0]:02d}.npz", **arrays)
        (root / f"figure_{FIG[0]:02d}.json").write_text(json.dumps(metadata, indent=2))
    _savefig(fig, os.path.join(_IMG, f"ConformalSquare_{FIG[0]:02d}.png"), dpi=72)

def _plot_cf(ax, cf, **kw):
    return matlab_plot(cf, ax=ax, **kw)

def _dot(ax, z):
    _source_plot_line_and_points(ax, np.real(z), np.imag(z), np.real(z), np.imag(z), '.r', {'markersize':20})

def run():
    os.makedirs(_IMG, exist_ok=True)
    print("@@CHEBFUNJAX_OUTPUT_SLOT:1@@", flush=True)
    z = cj.chebfun('z', domain=(0., 1.))
    w = _capture(fprime(z).cumsum(), 'initial_w')
    print("w =")
    print(repr(w), flush=True)
    print("@@CHEBFUNJAX_OUTPUT_SLOT:0@@", flush=True)
    zcirc = (1j * cj.chebfun('t', domain=(0., 2 * np.pi))).exp()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(600/72,270/72), dpi=72)
    # Native R2024b CPU/Painters subplot geometry at the source figure size.
    axL.set_position([0.13, 0.11, 0.33465909090909091, 0.815])
    axR.set_position([0.57034090909090907, 0.11, 0.33465909090909096, 0.815])
    _plot_cf(axL, zcirc, color='k', linewidth=2)
    _dot(axL, ZPRE)
    axL.set_aspect('equal', adjustable='box')
    axL.axis(1.05 * np.array([-1, 1, -1, 1]))
    axL.set_axis_off()
    matlab_plot(z.real(), z.imag(), ax=axL, color='b', linewidth=1.5)
    _plot_cf(axR, w, color='b', linewidth=1.5)
    axR.set_aspect('equal', adjustable='box')
    axR.axis(1.35 * np.array([-1, 1, -1, 1]))
    axR.set_axis_off()
    _snapshot(fig)
    print("@@CHEBFUNJAX_OUTPUT_SLOT:2@@", flush=True)
    for t in np.linspace(0, 2*np.pi, 33):
        e = np.exp(1j*t)
        z = cj.chebfun('r', domain=(0.,1.)) * e
        matlab_plot(z.real(), z.imag(), ax=axL, color='b', linewidth=1.5)
        ray = _capture((fprime(z) * e).cumsum(), f"ray_{round(t/(2*np.pi)*32):02d}")
        _plot_cf(axR, ray, color='b', linewidth=1.5)
    print("@@CHEBFUNJAX_OUTPUT_SLOT:0@@", flush=True)
    _snapshot(fig)
    corners = np.asarray(w(1)) * ZPRE[[0,1,2,3,0]]
    axR.plot(corners.real, corners.imag, 'k--', linewidth=2)
    _dot(axR, corners[:4])
    _snapshot(fig)
    for r in [.5,.6,.7,.8,.9,.97]:
        z = r * zcirc
        f = _capture(w(r) + (fprime(z) * z.diff()).cumsum(), f'circle_{r:.2f}')
        _plot_cf(axL, z, color='b', linewidth=1.5)
        _plot_cf(axR, f, color='b', linewidth=1.5)
    _snapshot(fig)
    print("@@CHEBFUNJAX_OUTPUT_SLOT:3@@", flush=True)
    f = _capture(w(1) + (fprime(zcirc) * zcirc.diff()).cumsum(), 'boundary_f')
    _plot_cf(axR, f, color='k', linewidth=2)
    _snapshot(fig)
    plt.close(fig)
    print("@@CHEBFUNJAX_OUTPUT_SLOT:0@@")
if __name__ == '__main__':
    run()
