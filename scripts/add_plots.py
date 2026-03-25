"""Script that injects matplotlib plotting code into all 50 example scripts.

Run from repo root:
    python scripts/add_plots.py
"""
from __future__ import annotations
import os
import re

# ---------------------------------------------------------------------------
# Each entry: (relative path from examples/, import_addition, plot_code)
# plot_code is inserted just before "    print("\nAll assertions passed.")"
# ---------------------------------------------------------------------------

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")


def _patch(rel_path: str, extra_imports: str, plot_code: str) -> None:
    """Read, patch and overwrite one example file."""
    full = os.path.join(EXAMPLES_DIR, rel_path)
    with open(full) as fh:
        src = fh.read()

    # 1. Add matplotlib import + plotting imports after "import chebfunjax as cj"
    mpl_header = (
        "import matplotlib\n"
        "matplotlib.use(\"Agg\")\n"
        "import matplotlib.pyplot as plt\n"
    )
    # Only inject if not already present
    if "matplotlib.use" not in src:
        # Find the first import line in the file
        src = re.sub(
            r"^(import jax\.numpy as jnp)",
            mpl_header + r"\1",
            src,
            count=1,
            flags=re.MULTILINE,
        )

    if extra_imports and extra_imports.strip() not in src:
        src = re.sub(
            r"^(import chebfunjax as cj)",
            r"\1\n" + extra_imports,
            src,
            count=1,
            flags=re.MULTILINE,
        )

    # 2. Insert plot_code before the sentinel line
    sentinel = '    print("\\nAll assertions passed.")'
    if sentinel in src and plot_code and plot_code.strip() not in src:
        src = src.replace(sentinel, plot_code + "\n" + sentinel)

    with open(full, "w") as fh:
        fh.write(src)
    print(f"  patched: {rel_path}")


# ---------------------------------------------------------------------------
# approx
# ---------------------------------------------------------------------------

_patch(
    "approx/absolute_value_newton.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(r, title="Newton iteration converging to |x|", label="r (converged)")
    _abs_cheb = cj.chebfun(lambda t: jnp.abs(t), domain=(-1.0, 0.0, 1.0))
    import numpy as _np
    _xs = _np.linspace(-1.0, 1.0, 600)
    ax.plot(_xs, _np.abs(_xs), "--", color="#E04040", linewidth=1.2, label="|x|")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "absolute_value_newton.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "approx/bessel_approximation.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(J0, title="Bessel functions on [0, 20]", label="J₀(x)")
    plot(J1, ax=ax, color="#E04040", label="J₁(x)")
    ax.axhline(0, color="k", linewidth=0.6)
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "bessel_approximation.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "approx/hermite_interpolation.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f_smooth, title="Polyfit of exp(sin(3x))", label="full chebfun")
    plot(p5, ax=ax, color="#E04040", linestyle="--", label="deg-5 polyfit")
    plot(p10, ax=ax, color="#228B22", linestyle=":", label="deg-10 polyfit")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "hermite_interpolation.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "approx/piecewise_smooth.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f_abs, title="Piecewise smooth functions", label="|x|")
    plot(f_hat, ax=ax, color="#E04040", label="1-|x|")
    plot(f_piecewise, ax=ax, color="#228B22", label="x² / sin(x)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "piecewise_smooth.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "approx/rational_like_convergence.py",
    "from chebfunjax.plotting import plot, plotcoeffs",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    _f1 = cj.chebfun(lambda x: 1.0 / (1.0 + x**2))
    _f25 = cj.chebfun(lambda x: 1.0 / (1.0 + 25.0 * x**2))
    fig, ax = plot(_f1, title="Convergence: functions near poles", label="1/(1+x²)")
    plot(_f25, ax=ax, color="#E04040", label="1/(1+25x²)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "rational_like_convergence.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plotcoeffs(_f25, title="|Chebyshev coeffs| of 1/(1+25x²)")
    fig2.savefig(os.path.join(_here, "rational_like_convergence_coeffs.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)
""",
)

_patch(
    "approx/special_functions.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f_erf, title="Special functions: erf and erfc", label="erf(x)")
    plot(f_erfc, ax=ax, color="#E04040", label="erfc(x)")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "special_functions.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plot(f_airy, title="Airy function Ai(x)")
    ax2.axhline(0, color="k", linewidth=0.5)
    fig2.savefig(os.path.join(_here, "special_functions_airy.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)
""",
)

# ---------------------------------------------------------------------------
# approx2
# ---------------------------------------------------------------------------

_patch(
    "approx2/smooth_functions_2d.py",
    "from chebfunjax.plotting import surf, contour",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = surf(f1, title="exp(x+y) on [-1,1]²")
    fig.savefig(os.path.join(_here, "smooth_functions_2d_exp.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = contour(f2, title="cos(x + y²) on [-1,1]²")
    fig2.savefig(os.path.join(_here, "smooth_functions_2d_cos.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = surf(f3, title="Franke's function on [-1,1]²")
    fig3.savefig(os.path.join(_here, "smooth_functions_2d_franke.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig3)
""",
)

_patch(
    "approx2/rank_of_functions.py",
    "from chebfunjax.plotting import surf, contour",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = surf(f1, title="cos(x)·exp(y) on [-1,1]²")
    fig.savefig(os.path.join(_here, "rank_of_functions.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig2, ax2 = contour(f2, title="sin(x)·sin(y) on [-1,1]²")
    fig2.savefig(os.path.join(_here, "rank_of_functions_contour.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)
""",
)

_patch(
    "approx2/integration_2d.py",
    "from chebfunjax.plotting import surf, contour",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = surf(f1, title="exp(x+y): 2-D integration")
    fig.savefig(os.path.join(_here, "integration_2d.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig2, ax2 = contour(f3, title="sin(πx)·sin(πy)")
    fig2.savefig(os.path.join(_here, "integration_2d_contour.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)
""",
)

_patch(
    "approx2/differentiation_2d.py",
    "from chebfunjax.plotting import surf, contour",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = surf(f1, title="exp(x+y): 2-D differentiation")
    fig.savefig(os.path.join(_here, "differentiation_2d.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig2, ax2 = contour(f2, title="x²y + sin(x)")
    fig2.savefig(os.path.join(_here, "differentiation_2d_f2.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)
""",
)

# ---------------------------------------------------------------------------
# calc
# ---------------------------------------------------------------------------

_patch(
    "calc/definite_indefinite_integrals.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f, title="2·cos(x) and its antiderivative on [0, 10]",
                   label="2·cos(x)")
    plot(g, ax=ax, color="#E04040", label="cumsum (antiderivative)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "definite_indefinite_integrals.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "calc/differentiation.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f, title="sin(x) and its derivative", label="sin(x)")
    plot(df, ax=ax, color="#E04040", label="cos(x)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "differentiation.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "calc/bird_flight_optimization.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(T, title="Bird-flight travel time T(x)", ylabel="T")
    ax.axvline(float(x_min), color="#E04040", linewidth=1.2,
               linestyle="--", label=f"x* = {float(x_min):.4f}")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "bird_flight_optimization.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "calc/mean_value_theorem.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f, title="Mean Value Theorem: sin(x) on [0, 2π]",
                   label="sin(x)")
    plot(df, ax=ax, color="#E04040", label="cos(x) = f′(x)")
    ax.axhline(mean_slope, color="#228B22", linewidth=1.2,
               linestyle="--", label=f"mean slope = {mean_slope:.2f}")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "mean_value_theorem.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "calc/snells_law.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(T, title="Snell's law: travel time T(x)", ylabel="T")
    ax.axvline(float(x_min), color="#E04040", linewidth=1.2,
               linestyle="--", label=f"x* = {float(x_min):.4f}")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "snells_law.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

# ---------------------------------------------------------------------------
# roots
# ---------------------------------------------------------------------------

_patch(
    "roots/bessel_roots.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(J0, title="J₀(x) and its zeros on [0, 100]")
    ax.axhline(0, color="k", linewidth=0.5)
    _r_arr = r_arr[:30]  # first 30 roots
    import numpy as _np
    ax.plot(_r_arr, _np.zeros_like(_r_arr), "o", color="#E04040",
            markersize=3, label="roots")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "bessel_roots.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "roots/extrema_and_roots.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f, title="sin(x): roots and extrema on [0, 2π]")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.plot(float(x_min), float(y_min), "v", color="#E04040",
            markersize=8, label="min")
    ax.plot(float(x_max), float(y_max), "^", color="#228B22",
            markersize=8, label="max")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "extrema_and_roots.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "roots/newton_raphson.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f, title="x³ − 3x² + 2: roots", label="p(x)")
    ax.axhline(0, color="k", linewidth=0.6)
    import numpy as _np
    ax.plot(r_arr, _np.zeros_like(r_arr), "o", color="#E04040",
            markersize=7, label="roots")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "newton_raphson.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "roots/polynomial_roots.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f_T2, title="Chebyshev polynomial roots", label="T₂(x)")
    plot(f_T4, ax=ax, color="#E04040", label="T₄(x)")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "polynomial_roots.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "roots/random_polynomials.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    import matplotlib.pyplot as _plt
    import numpy as _np
    fig, ax = _plt.subplots(figsize=(6, 3.5))
    _f_plot = Chebfun(funs=[piece], domain=Domain((-1.0, 1.0)))
    _xs = _np.linspace(-1.0, 1.0, 600)
    ax.plot(_xs, _np.array(_f_plot(jnp.array(_xs))), color="#4169E1",
            linewidth=1.5, label="random poly (n=15)")
    _r2 = _np.array(_f_plot.roots())
    ax.plot(_r2, _np.zeros_like(_r2), "ro", markersize=5, label="roots")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_title("Random Chebyshev polynomial roots", fontsize=11)
    ax.set_xlabel("x", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "random_polynomials.png"),
                dpi=150, bbox_inches="tight")
    _plt.close(fig)
""",
)

# ---------------------------------------------------------------------------
# quad
# ---------------------------------------------------------------------------

_patch(
    "quad/clenshaw_curtis.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    import matplotlib.pyplot as _plt
    import numpy as _np
    fig, ax = _plt.subplots(figsize=(6, 3.5))
    _funcs = [
        ("exp(x)",   lambda x: jnp.exp(x),       (-1.0, 1.0)),
        ("sin(x)",   lambda x: jnp.sin(x),        (0.0, float(jnp.pi))),
        ("exp(-x²)", lambda x: jnp.exp(-x**2),   (-5.0, 5.0)),
    ]
    _colors = ["#4169E1", "#E04040", "#228B22"]
    for (_name, _fn, _dom), _col in zip(_funcs, _colors):
        _f = cj.chebfun(_fn, domain=_dom)
        _xs = _np.linspace(float(_dom[0]), float(_dom[1]), 400)
        ax.plot(_xs, _np.array(_f(jnp.array(_xs))), color=_col,
                linewidth=1.5, label=_name)
    ax.set_title("Clenshaw-Curtis integrands", fontsize=11)
    ax.set_xlabel("x", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "clenshaw_curtis.png"),
                dpi=150, bbox_inches="tight")
    _plt.close(fig)
""",
)

_patch(
    "quad/gauss_quadrature.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    import matplotlib.pyplot as _plt
    import numpy as _np
    fig, ax = _plt.subplots(figsize=(6, 3.5))
    for _n, _col in [(5, "#4169E1"), (10, "#E04040"), (20, "#228B22")]:
        _nodes, _weights = golub_welsch(_n)
        ax.plot(_nodes, _np.zeros_like(_nodes), "o", color=_col,
                markersize=5, label=f"n={_n}")
    ax.set_title("Gauss-Legendre nodes on [-1, 1]", fontsize=11)
    ax.set_xlabel("x", fontsize=10)
    ax.set_yticks([])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "gauss_quadrature.png"),
                dpi=150, bbox_inches="tight")
    _plt.close(fig)
""",
)

_patch(
    "quad/convergence_rates.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f, title="exp(cos(x)) — smooth integrand")
    fig.savefig(os.path.join(_here, "convergence_rates.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "quad/tricky_integrals.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f1, title="sin(100x) — highly oscillatory")
    fig.savefig(os.path.join(_here, "tricky_integrals.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

# ---------------------------------------------------------------------------
# ode-linear
# ---------------------------------------------------------------------------

_patch(
    "ode-linear/airy_equation.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(u, title="Airy equation: Ai(x) on [-10, 2]",
                   label="Chebfun solution")
    import numpy as _np; import scipy.special as _sp
    _xs = _np.linspace(a, b, 400)
    ax.plot(_xs, _sp.airy(_xs)[0], "--", color="#E04040",
            linewidth=1.2, label="scipy Ai(x)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "airy_equation.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "ode-linear/bessel_bvp.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(u1, title="Bessel BVP: y″ + y/x − y·ν²/x² = 0",
                   label="u₁ (ν=0)")
    plot(u2, ax=ax, color="#E04040", label="u₂ (ν=1)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "bessel_bvp.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "ode-linear/boundary_layer.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    import matplotlib.pyplot as _plt
    import numpy as _np
    fig, ax = _plt.subplots(figsize=(6, 3.5))
    _colors = ["#4169E1", "#E04040"]
    for _eps, _col in zip([0.1, 0.01], _colors):
        _exact = lambda x, e=_eps: (1.0 - _np.exp(-x / e)) / (1.0 - _np.exp(-1.0 / e))
        _xs = _np.linspace(0.0, 1.0, 500)
        ax.plot(_xs, _exact(_xs), color=_col, linewidth=1.5,
                label=f"ε = {_eps}")
    ax.set_title("Boundary layer: ε u″ + u′ = 0", fontsize=11)
    ax.set_xlabel("x", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "boundary_layer.png"),
                dpi=150, bbox_inches="tight")
    _plt.close(fig)
""",
)

_patch(
    "ode-linear/linear_ivp_cosine.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(u, title="u′ − u = 0, u(0)=1 (solution = exp(x))",
                   label="Chebfun")
    plot(cos_cheb, ax=ax, color="#E04040", linestyle="--", label="cos(x)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "linear_ivp_cosine.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "ode-linear/poisson_equation.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(u1, title="Poisson equation: −u″ = f on [0,1]",
                   label="u₁ (f=1)")
    plot(u2, ax=ax, color="#E04040", label="u₂ (f=π²sin(πx))")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "poisson_equation.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "ode-linear/wiki_odes.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(u1, title="Linear ODEs", label="u₁")
    plot(u2, ax=ax, color="#E04040", label="u₂")
    plot(u3, ax=ax, color="#228B22", label="u₃")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "wiki_odes.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

# ---------------------------------------------------------------------------
# ode-nonlin
# ---------------------------------------------------------------------------

_patch(
    "ode-nonlin/carrier_equation.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(u, title="Carrier equation: ε u″ + 2(1−x²)u + u² = 1")
    fig.savefig(os.path.join(_here, "carrier_equation.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "ode-nonlin/exact_solutions_bender_orszag.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(u1, title="Nonlinear BVP (Bender & Orszag)",
                   label="u₁")
    plot(u2, ax=ax, color="#E04040", label="u₂")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "exact_solutions_bender_orszag.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "ode-nonlin/logistic_equation.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(u, title="Logistic equation: u′ = u(1−u)",
                   label="u (standard)")
    plot(u2, ax=ax, color="#E04040", label="u₂ (modified)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "logistic_equation.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "ode-nonlin/pendulum_equation.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(theta, title="Pendulum equation: θ″ + sin θ = 0",
                   ylabel="θ (rad)")
    fig.savefig(os.path.join(_here, "pendulum_equation.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

# ---------------------------------------------------------------------------
# ode-eig
# ---------------------------------------------------------------------------

_patch(
    "ode-eig/double_well.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    import matplotlib.pyplot as _plt
    import numpy as _np
    fig, ax = _plt.subplots(figsize=(6, 3.5))
    _lam_real = _np.sort(_np.real(_np.array(lam[:n_eigs])))
    ax.bar(_np.arange(n_eigs), _lam_real, color="#4169E1", alpha=0.8)
    ax.set_xlabel("eigenvalue index", fontsize=10)
    ax.set_ylabel("λ", fontsize=10)
    ax.set_title("Double-well Schrödinger eigenvalues", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "double_well.png"),
                dpi=150, bbox_inches="tight")
    _plt.close(fig)
""",
)

_patch(
    "ode-eig/harmonic_oscillator.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    import matplotlib.pyplot as plt
    import numpy as _np
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(_np.arange(n_eigs), _np.array(lam[:n_eigs]), color="#4169E1",
           alpha=0.8, label="computed")
    ax.plot(_np.arange(n_eigs), exact, "ro", markersize=6, label="exact")
    ax.set_xlabel("eigenvalue index", fontsize=10)
    ax.set_ylabel("λ", fontsize=10)
    ax.set_title("Quantum harmonic oscillator eigenvalues", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "harmonic_oscillator.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "ode-eig/laplacian_eigenvalues.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    import matplotlib.pyplot as plt
    import numpy as _np
    fig, ax = plt.subplots(figsize=(6, 3.5))
    _n = _np.arange(1, k + 1)
    ax.plot(_n, _np.array(lam[:k]), "o", color="#4169E1",
            markersize=7, label="computed (Dirichlet)")
    ax.plot(_n, _n**2, "--", color="#E04040", linewidth=1.4,
            label="exact n²")
    ax.set_xlabel("n", fontsize=10)
    ax.set_ylabel("λ", fontsize=10)
    ax.set_title("Laplacian eigenvalues on [0, π]", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "laplacian_eigenvalues.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "ode-eig/sturm_liouville.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    import matplotlib.pyplot as _plt
    import numpy as _np
    fig, ax = _plt.subplots(figsize=(6, 3.5))
    _lam1_arr = _np.sort(_np.real(_np.array(lam1[:5])))
    ax.plot(_np.arange(1, 6), _lam1_arr, "o-", color="#4169E1",
            linewidth=1.5, markersize=6, label="−u″ on [0,π]")
    ax.set_xlabel("n", fontsize=10)
    ax.set_ylabel("λ", fontsize=10)
    ax.set_title("Sturm-Liouville eigenvalues", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "sturm_liouville.png"),
                dpi=150, bbox_inches="tight")
    _plt.close(fig)
""",
)

# ---------------------------------------------------------------------------
# opt
# ---------------------------------------------------------------------------

_patch(
    "opt/catenary.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(y, title="Catenary y = a·cosh(x/a)", ylabel="y")
    fig.savefig(os.path.join(_here, "catenary.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "opt/global_minimum_2d.py",
    "from chebfunjax.plotting import surf, contour",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = contour(f1, title="(x−0.3)² + (y+0.5)² — global minimum")
    ax.plot(0.3, -0.5, "r*", markersize=12, label="min (0.3, −0.5)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "global_minimum_2d.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "opt/minimum_of_smooth_function.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f, title="x² + sin(x): global minimum on [−π, π]")
    ax.plot(float(x_min), float(y_min), "r*", markersize=12, label="min")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "minimum_of_smooth_function.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plot(g, title="sin(8x)+sin(5x)+0.3x on [0, 2π]")
    fig2.savefig(os.path.join(_here, "minimum_of_smooth_function2.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)
""",
)

# ---------------------------------------------------------------------------
# linalg
# ---------------------------------------------------------------------------

_patch(
    "linalg/chebfun_inner_products.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f, title="Inner products: sin(πx) and cos(πx)",
                   label="sin(πx)")
    plot(g, ax=ax, color="#E04040", label="cos(πx)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "chebfun_inner_products.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "linalg/matrix_functions.py",
    "from chebfunjax.plotting import plot, plotcoeffs",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plotcoeffs(f_exp,
                         title="|Chebyshev coeffs| of exp(x)")
    fig.savefig(os.path.join(_here, "matrix_functions.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "linalg/resolvent_norm.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(u1, title="Eigenfunctions of −d²/dx² on [0,1]",
                   label="sin(πx)")
    plot(u1p, ax=ax, color="#E04040", label="π·cos(πx)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "resolvent_norm.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

# ---------------------------------------------------------------------------
# complex
# ---------------------------------------------------------------------------

_patch(
    "complex/argument_principle.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(p1, title="(x−1)(x−2)(x−3): roots on [0,4]",
                   label="p(x)")
    ax.axhline(0, color="k", linewidth=0.5)
    import numpy as _np
    ax.plot(roots1_arr, _np.zeros_like(roots1_arr), "r^",
            markersize=8, label="roots")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "argument_principle.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

_patch(
    "complex/contour_integrals.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f_re, title="Cauchy integral: Re and Im parts of integrand",
                   label="Re part")
    plot(f_im, ax=ax, color="#E04040", label="Im part")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "contour_integrals.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
""",
)

# ---------------------------------------------------------------------------
# fourier
# ---------------------------------------------------------------------------

_patch(
    "fourier/fourier_coefficients.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    import matplotlib.pyplot as _plt
    import numpy as _np
    # Plot |Fourier coefficients| of cos(3x) and sawtooth
    fig, axes = _plt.subplots(1, 2, figsize=(9, 3.5))
    _ns = _np.arange(0, 10)
    _cos3x_an = _np.array([abs(compute_fourier_coeff(lambda x: jnp.cos(3.0*x), n, dom)[0])
                            for n in _ns])
    axes[0].bar(_ns, _cos3x_an, color="#4169E1", alpha=0.8)
    axes[0].set_title("|a_n| of cos(3x)", fontsize=11)
    axes[0].set_xlabel("n", fontsize=10)
    _saw_bn = _np.array([abs(compute_fourier_coeff(lambda x: x, n, dom)[1])
                          for n in range(1, 11)])
    axes[1].bar(_np.arange(1, 11), _saw_bn, color="#E04040", alpha=0.8)
    axes[1].set_title("|b_n| of sawtooth f(x)=x", fontsize=11)
    axes[1].set_xlabel("n", fontsize=10)
    for _ax in axes:
        _ax.grid(True, alpha=0.3, linestyle="--")
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "fourier_coefficients.png"),
                dpi=150, bbox_inches="tight")
    _plt.close(fig)
""",
)

_patch(
    "fourier/gibbs_phenomenon.py",
    "from chebfunjax.plotting import plot",
    """\
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    import matplotlib.pyplot as _plt
    import numpy as _np
    _x = _np.linspace(0.0, 2.0 * _np.pi, 1000)
    fig, ax = _plt.subplots(figsize=(6, 3.5))
    ax.step(_x, _np.where(_x < _np.pi, -1.0, 1.0), color="k",
            linewidth=0.8, label="sign(x−π)")
    for _N, _col in [(5, "#4169E1"), (20, "#E04040"), (50, "#228B22")]:
        ax.plot(_x, partial_fourier_sum(_N, _x), color=_col,
                linewidth=1.2, label=f"N={_N}")
    ax.legend(fontsize=9)
    ax.set_xlabel("x", fontsize=10)
    ax.set_title("Gibbs phenomenon", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "gibbs_phenomenon.png"),
                dpi=150, bbox_inches="tight")
    _plt.close(fig)
""",
)

print("Done patching all examples.")
