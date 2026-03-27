"""Global optimization: the Rosenbrock function.

Minimizes the Rosenbrock function f(x,y) = (1-x)^2 + 100*(y-x^2)^2
by taking slices with 1D Chebfun. Based on Chebfun example opt/Rosenbrock.m
by Nick Trefethen (October 2010).

Original: https://www.chebfun.org/examples/opt/Rosenbrock.html
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/opt')
    os.makedirs(outdir, exist_ok=True)

    # Rosenbrock function: minimum at (1, 1) with value 0
    def rosenbrock(x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2

    # For each x0, find min over y in [-1, 3]
    def fmin_at_x0(x0):
        """Min of f(x0, y) over y in [-1, 3]."""
        f_y = cj.chebfun(lambda y: (1.0 - x0)**2 + 100.0 * (y - x0**2)**2,
                         domain=(-1.0, 3.0))
        _, min_val = f_y.min()
        return float(min_val)

    # Build fminx as a function of x in [-1.5, 1.5]
    x_vals = np.linspace(-1.5, 1.5, 40)
    fmin_vals = np.array([fmin_at_x0(x0) for x0 in x_vals])

    # Find global minimum
    idx_min = np.argmin(fmin_vals)
    x_min = x_vals[idx_min]
    print(f"Approx x* from grid: {x_min:.6f}  (exact: 1.0)")

    # Refine: build fminx on a fine grid and use chebfun interpolation
    x_fine = np.linspace(-1.5, 1.5, 100)
    fmin_fine = np.array([fmin_at_x0(x0) for x0 in x_fine])
    idx_min2 = np.argmin(fmin_fine)
    x_opt = x_fine[idx_min2]
    min_f = fmin_fine[idx_min2]
    print(f"Minimum of Rosenbrock: f* ≈ {min_f:.2e}  at x* ≈ {x_opt:.6f}")
    print(f"Exact minimum: 0.0  at (1.0, 1.0)")

    # Find y* given x*
    f_y_at_xstar = cj.chebfun(lambda y: (1.0 - float(x_opt))**2 + 100.0 * (y - float(x_opt)**2)**2,
                               domain=(-1.0, 3.0))
    y_opt, _ = f_y_at_xstar.min()
    print(f"y* = {y_opt:.10f}  (exact: 1.0)")

    # Plot
    fig, axes = plt.subplots(1, 2)

    # Contour plot
    x = np.linspace(-1.5, 1.5, 200)
    y = np.linspace(-1.0, 3.0, 200)
    XX, YY = np.meshgrid(x, y)
    ZZ = rosenbrock(XX, YY)
    axes[0].contour(x, y, ZZ, levels=np.arange(10, 301, 20), colors='gray',
                    linewidths=0.6)
    axes[0].contourf(x, y, ZZ, levels=np.arange(0, 301, 30),
                     cmap='Blues_r', alpha=0.6)
    axes[0].plot(x_opt, y_opt, 'r*', markersize=12, label=f'min at ({x_opt:.3f},{y_opt:.3f})')
    axes[0].plot(1.0, 1.0, 'k+', markersize=12, linewidth=2, label='True min (1,1)')
    axes[0].set_title('Rosenbrock function $f(x,y)$', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].set_aspect('equal')

    # Min over y slices
    axes[1].plot(x_fine, fmin_fine, 'b.-', markersize=4, linewidth=1.2,
                 label='$\\min_y f(x,y)$')
    axes[1].axvline(x_opt, color='r', linestyle='--', linewidth=1.2,
                    label=f'$x^* \\approx {x_opt:.3f}$')
    axes[1].set_title('$\\min_y f(x,y)$ — finding the global minimum', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].set_yscale('symlog', linthresh=1e-2)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'rosenbrock.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    assert abs(x_opt - 1.0) < 0.05, f"x* = {x_opt} (expected ~1.0)"
    assert abs(y_opt - 1.0) < 0.05, f"y* = {y_opt} (expected ~1.0)"
    assert min_f < 0.01, f"f* = {min_f} (expected ~0)"

    print("rosenbrock: done")
    return True

if __name__ == "__main__":
    run()
