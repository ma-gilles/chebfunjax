"""Native trigremez iteration presentation, independent of exchange numerics.

Provenance
----------
MATLAB source: @chebfun/trigremez.m, doPlotIter and doDisplayIter.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
"""

import jax.numpy as jnp

MATLAB_COLOR_ORDER = (
    (0.0, 0.447, 0.741),
    (0.85, 0.325, 0.098),
    (0.929, 0.694, 0.125),
    (0.494, 0.184, 0.556),
    (0.466, 0.674, 0.188),
    (0.301, 0.745, 0.933),
    (0.635, 0.078, 0.184),
)

HEADER = "It.     Max(|Error|)       |ErrorRef|      Delta ErrorRef      Delta Ref"


def iteration_hook(display, plot, *, ax=None, hold=False):
    """Construct the optional presentation callback; absent flags do nothing.

    Python exposes native current axes and hold state as ``ax`` and ``hold``.
    Neither presentation path updates the numerical exchange state.
    """
    if not display and not plot:
        return None
    if display:
        print(HEADER)

    def callback(iteration, error, h, delta, normf, diffx, old, new, target, trial, domain):
        if plot:
            plot_iteration(old, new, lambda x: target(x) - trial(x), domain, ax=ax, hold=hold)
        if display:
            print(
                f"{iteration}        {float(error):5.4e}        {float(jnp.abs(h)):5.4e}"
                f"        {float(delta / normf):5.4e}        {float(diffx):5.4e}"
            )

    return callback


def plot_iteration(old, new, error, domain, *, ax=None, hold=False):
    """Render native doPlotIter's discrete references and exact error samples.

    Default native graphics properties are qualified separately from solver
    arithmetic. A supplied axes keeps its geometry; ``hold=True`` keeps its
    earlier artists. The callback restores that same hold policy each call.
    """
    import matplotlib.pyplot as plt
    from matplotlib.path import Path

    from chebfunjax.plotting import matlab_plot

    if ax is None:
        ax = plt.gca()
    if not hold:
        ax.clear()
        ax.set_prop_cycle(color=MATLAB_COLOR_ORDER)
    # The native first marker line consumes a color-order slot; matlab_plot
    # does not advance for its explicit LineSpec color. Preserve the held cycle.
    ax._get_lines.get_next_color()
    a, b = domain

    def forward(y):
        return b * (y + jnp.pi) / (2 * jnp.pi) + a * (jnp.pi - y) / (2 * jnp.pi)

    old, new = forward(old), forward(new)
    matlab_plot(
        old,
        jnp.zeros_like(old),
        "or",
        ax=ax,
        markersize=12,
        linewidth=0.5,
        markerfacecolor="none",
        markeredgewidth=0.5,
    )
    # The black reference uses a named Color property here, so matlab_plot
    # performs its native color-order advance internally.
    diagonal = float(0.5 / jnp.sqrt(2.0))
    star = Path(
        [
            (0, -0.5),
            (0, 0.5),
            (-0.5, 0),
            (0.5, 0),
            (-diagonal, -diagonal),
            (diagonal, diagonal),
            (-diagonal, diagonal),
            (diagonal, -diagonal),
        ],
        [Path.MOVETO, Path.LINETO] * 4,
    )
    matlab_plot(
        new,
        jnp.zeros_like(new),
        ax=ax,
        color="black",
        marker=star,
        linestyle="none",
        markersize=12,
        linewidth=0.5,
        markerfacecolor="none",
        markeredgewidth=0.5,
    )
    xx = jnp.linspace(a, b, 5000)
    yy = jnp.linspace(-jnp.pi, jnp.pi, 5000)
    matlab_plot(xx, error(yy), ax=ax, linewidth=0.5)
    ax.set_xlim(a, b)
    ax.legend(["Current Ref.", "Next Ref.", "Error"])
    ax.figure.canvas.draw_idle()
    return ax


OPTION_OMITTED = object()


def parse_options(m, n, positional, max_iter, tol, display, plotfcns):
    """Adapt native case-insensitive pairs and prior numeric Python positions.

    Source display/plotfcns enable the flag whenever the option is present;
    their supplied value is deliberately ignored. Later native pairs win.
    """
    from chebfunjax.utils.trigrational import TrigRemezError

    show = display is not OPTION_OMITTED
    plot = plotfcns is not OPTION_OMITTED
    if isinstance(n, (str, bool)):
        options = (n, *positional)
        n = None
    else:
        options = positional
        # Preserve the former Python fourth/fifth positional arguments.
        if options and all(not isinstance(value, str) for value in options) and len(options) <= 2:
            max_iter = options[0]
            if len(options) == 2:
                tol = options[1]
            options = ()
    for degree in (m, 0 if n is None else n):
        if degree != degree or degree < 0 or degree != round(degree):
            raise TrigRemezError(
                "CHEBFUN:CHEBFUN:trigremez:parseInputs",
                "Degree of approximation must be a nonnegative integer.",
            )
    for index in range(0, len(options), 2):
        name = options[index]
        if not isinstance(name, str) or name.lower() not in {
            "tol",
            "maxiter",
            "display",
            "plotfcns",
        }:
            raise TrigRemezError(
                "CHEBFUN:CHEBFUN:trigremez:badInput", "Unrecognized sequence of input parameters."
            )
        if name.lower() == "display":
            show = True
        elif name.lower() == "plotfcns":
            plot = True
        else:
            if index + 1 >= len(options):
                raise TrigRemezError(
                    "MATLAB:badsubscript",
                    "Index exceeds the number of array elements. "
                    f"Index must not exceed {len(options)}.",
                )
            value = options[index + 1]
            if name.lower() == "tol":
                tol = value
            else:
                max_iter = value
    return n, max_iter, tol, show, plot
