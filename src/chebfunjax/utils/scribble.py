"""Write text as a complex-valued piecewise-linear Chebfun.

Translated from MATLAB Chebfun scribble.m (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun
Developers. See https://www.chebfun.org/ for Chebfun information.

Note: PLAN §10 originally deferred scribble as a "visualization toy",
but it gates seven Guide-chapter-5 figures and several examples, and
the stroke font translates directly, so it is included. The stroke
coordinates below are copied verbatim from the MATLAB source.
"""

from __future__ import annotations

import warnings

# Stroke font: each character is a list of subpaths; each subpath is a
# list of complex vertices in the unit em-box, joined by line segments.
# Copied verbatim from MATLAB scribble.m.
_F = {
    "A": [[0, .4 + 1j, .8, .6 + .5j, .2 + .5j]],
    "B": [[0, 1j, .8 + .9j, .8 + .6j, .5j, .8 + .4j, .8 + .1j, 0]],
    "C": [[.8 + 1j, .8j, .2j, .8]],
    "D": [[0, .8 + .1j, .8 + .9j, 1j, 0]],
    "E": [[.8 + 1j, 1j, .5j, .5j + .7, .5j, 0, .8]],
    "F": [[.8 + 1j, 1j, .5j, .5j + .7, .5j, 0]],
    "G": [[.8 + 1j, .8j, .2j, .6, .6 + .5j, .4 + .5j, .8 + .5j]],
    "H": [[0, 1j, .5j, .5j + .8, .8 + 1j, .8]],
    "I": [[0, .8, .4, .4 + 1j, 1j, .8 + 1j]],
    "J": [[0, .4, .4 + 1j, 1j, .8 + 1j]],
    "K": [[0, 1j, .5j, .8 + 1j, .5j, .8]],
    "L": [[1j, 0, .8]],
    "M": [[0, .1 + 1j, .4, .7 + 1j, .8]],
    "N": [[0, 1j, .8, .8 + 1j]],
    "O": [[0, 1j, .8 + 1j, .8, 0]],
    "P": [[0, 1j, .8 + 1j, .8 + .5j, .5j]],
    "Q": [[0, 1j, .8 + 1j, .8, .6 + .2j, .9 - .1j, .8, 0]],
    "R": [[0, 1j, .8 + 1j, .8 + .6j, .5j, .8]],
    "S": [[.8 + 1j, .9j, .6j, .8 + .4j, .8 + .1j, 0]],
    "T": [[.4, .4 + 1j, 1j, .8 + 1j]],
    "U": [[1j, .1, .7, .8 + 1j]],
    "V": [[1j, .4, .8 + 1j]],
    "W": [[1j, .2, .4 + 1j, .6, .8 + 1j]],
    "X": [[1j, .8, .4 + .5j, .8 + 1j, 0]],
    "Y": [[1j, .4 + .5j, .8 + 1j, .4 + .5j, .4]],
    "Z": [[1j, .8 + 1j, 0, .8]],
    "0": [[0, 1j, .8 + 1j, 0, .8, .8 + 1j]],
    "1": [[0, .8, .4, .4 + 1j, .1 + .8j]],
    "2": [[.8, 0, .5j, .8 + .5j, .8 + 1j, 1j]],
    "3": [[1j, .8 + 1j, .8 + .5j, .1 + .5j, .8 + .5j, .8, 0]],
    "4": [[1j, .5j, .8 + .5j, .8 + 1j, .8]],
    "5": [[.8 + 1j, 1j, .5j, .8 + .5j, .8, 0]],
    "6": [[.8 + 1j, 1j, 0, .8, .8 + .5j, .5j]],
    "7": [[1j, .8 + 1j, .2]],
    "8": [[1j, .8 + 1j, .8, 0, .5j, 1j, .5j, .8 + .5j]],
    "9": [[.8, .8 + 1j, 1j, .5j, .8 + .5j]],
    ".": [[0, .05, .05 + .05j, .05j, 0]],
    ",": [[-0.1 - .15j, -.05 - .15j, .1 + .05j, .05 + .05j, -.1 - .15j]],
    "?": [[.6j, 1j, .8 + 1j, .8 + .5j, .4 + .5j, .4 + .2j],
          [.35, .45, .45 + .05j, .35 + .05j, .35]],
    "!": [[.4 + 1j, .4 + .2j],
          [.35, .45, .45 + .05j, .35 + .05j, .35]],
    "'": [[.3 + .7j, .4 + 1j]],
    "-": [[.1 + .5j, .7 + .5j]],
    "+": [[.5j, .8 + .5j, .4 + .5j, .4 + .9j, .4 + .1j]],
    "/": [[.2, .6 + 1j]],
    "=": [[.1 + .4j, .7 + .4j], [.1 + .6j, .7 + .6j]],
    " ": [],
}


def scribble(s: str, dom=(-1.0, 1.0)):
    """Write text with a complex-valued piecewise-linear Chebfun.

    ``scribble(s)`` returns a complex Chebfun parameterised on
    ``[-1, 1]`` whose image draws the text ``s`` (upper-cased) inside
    the box ``[-1, 1] x [0, 2/len(s)]`` of the complex plane, exactly
    like MATLAB's ``scribble``.

    Parameters
    ----------
    s : str
        The text. Characters without a stroke definition draw nothing
        (with a warning).
    dom : (float, float), optional
        Parameter interval, default ``(-1, 1)``.

    Returns
    -------
    Chebfun
        Piecewise-linear complex Chebfun with one piece per stroke.

    Examples
    --------
    >>> f = scribble('CHEBFUN')
    >>> f.plot()  # doctest: +SKIP

    Provenance
    ----------
    MATLAB source : scribble.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
    from chebfunjax.domain import Domain

    n_chars = max(len(s), 1)
    scale = 2.0 / n_chars

    # Collect stroke endpoint pairs in final text coordinates:
    # vertex v of character j maps to (v + j)*2/len(s) - 1.
    strokes: list[tuple[complex, complex]] = []
    for j, ch in enumerate(s):
        paths = _F.get(ch.upper())
        if paths is None:
            warnings.warn(
                f'"{ch}" is not supported by scribble.', stacklevel=2
            )
            continue
        for path in paths:
            pts = [(complex(v) + j) * scale - 1.0 for v in path]
            strokes.extend(zip(pts[:-1], pts[1:]))

    if not strokes:
        raise ValueError("scribble: no drawable characters in input.")

    ns = len(strokes)
    a, b = float(dom[0]), float(dom[1])
    h = (b - a) / ns

    # One exactly-linear complex piece per stroke, built directly on the
    # final domain (MATLAB builds on [0, ns] then remaps; same result).
    pieces = []
    breakpoints = [a + k * h for k in range(ns + 1)]
    breakpoints[-1] = b
    for k, (w0, w1) in enumerate(strokes):
        t0, t1 = breakpoints[k], breakpoints[k + 1]
        slope = (w1 - w0) / (t1 - t0)

        def seg(t, _w0=w0, _s=slope, _t0=t0):
            return _w0 + _s * (t - _t0)

        pieces.append(_Piece.from_function(seg, t0, t1, n=2))

    return Chebfun(funs=pieces, domain=Domain(tuple(breakpoints)))
