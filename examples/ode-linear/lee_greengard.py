"""Lee and Greengard stiff ODE examples.

Solves three benchmark problems from Lee & Greengard (1997) using
spectral methods, verifying that Chebyshev collocation handles moderate stiffness.

Examples:
  1. Viscous shock: eps u'' + 2x u' = 0, u(±1) = ±1
  2. Turning point: eps u'' + x u' - u = 0, u(0)=0, u(1)=1
  3. Interior layer: eps u'' - (x-.3) u = 0, u(-1)=u(1)=0 + particular solution

Credit: Chebfun example ode-linear/LeeGreengardODEs.m (Nick Trefethen, Jun 2012).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
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

from chebfunjax.operators.chebop import Chebop

def run():
    print("=" * 60)
    print("Lee and Greengard stiff ODE examples")
    print("=" * 60)

    # Example 1: Viscous shock
    # eps u'' + 2x u' = 0, u(-1)=-1, u(1)=1
    # Exact: u = erf(x / sqrt(eps)) / erf(1 / sqrt(eps))
    print("\nExample 1: Viscous shock  eps u'' + 2x u' = 0, u(±1) = ±1")
    from scipy.special import erf

    dom = (-1.0, 1.0)
    eps = 0.01
    exact_shock = lambda x, eps: erf(x / np.sqrt(eps)) / erf(1.0 / np.sqrt(eps))

    N1 = Chebop(lambda x, u: eps * u.diff(2) + 2.0 * x * u.diff(), domain=dom)
    N1.lbc = -1.0
    N1.rbc = 1.0
    u1 = N1.solve(0.0)
    x_test = jnp.linspace(-1.0, 1.0, 400)
    err1 = float(jnp.max(jnp.abs(u1(x_test) -
                                  jnp.array(exact_shock(np.array(x_test), eps)))))
    print(f"  eps={eps}, max error: {err1:.2e}")
    assert err1 < 1e-8

    # Example 2: Interior layer near x=0.3
    # eps u'' + (x - 0.3) u' = 0, u(-1)=0, u(1)=1
    print("\nExample 2: Interior layer  eps u'' + (x-0.3) u' = 0, u(-1)=0, u(1)=1")
    eps2 = 0.01
    N2 = Chebop(
        lambda x, u: eps2 * u.diff(2) + (x - 0.3) * u.diff(),
        domain=dom
    )
    N2.lbc = 0.0
    N2.rbc = 1.0
    u2 = N2.solve(0.0)
    print(f"  eps={eps2}, solution length: {len(u2)}")
    assert abs(float(u2(jnp.array(-1.0)))) < 1e-8
    assert abs(float(u2(jnp.array(1.0))) - 1.0) < 1e-8
    # Maximum should occur near x=0.3 (interior layer)
    x_dense = jnp.linspace(-1.0, 1.0, 1000)
    u2_vals = u2(x_dense)
    idx_max_deriv = int(jnp.argmax(jnp.abs(u2.diff()(x_dense))))
    x_at_layer = float(x_dense[idx_max_deriv])
    print(f"  Steepest gradient at x ≈ {x_at_layer:.3f}  (near layer at x=0.3)")
    assert abs(x_at_layer - 0.3) < 0.1

    # Example 3: Airy-like equation
    print("\nExample 3: eps u'' - x u = 0  (Airy equation), u(-1)=Ai(-1/eps^(1/3)),...")
    from scipy.special import airy
    eps3 = 0.01
    # Exact solution: u = Ai(x / eps^(1/3)) / Ai(-1 / eps^(1/3))
    scale = eps3 ** (1.0/3.0)
    Ai_vals = lambda x: airy(x / scale)[0]
    rbc_val = float(Ai_vals(np.array(1.0))) / float(Ai_vals(np.array(-1.0)))

    N3 = Chebop(lambda x, u: eps3 * u.diff(2) - x * u, domain=dom)
    N3.lbc = 1.0    # u(-1) = 1 (normalize Ai(-1))
    N3.rbc = rbc_val
    u3 = N3.solve(0.0)
    exact3 = lambda x: np.array(Ai_vals(x)) / float(Ai_vals(np.array(-1.0)))
    x_test3 = np.linspace(-1.0, 1.0, 300)
    err3 = float(np.max(np.abs(np.array(u3(jnp.array(x_test3, dtype=jnp.float64))) -
                                exact3(x_test3))))
    print(f"  eps={eps3}, max error: {err3:.2e}")
    assert err3 < 1e-6

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 3)

    x_plot = jnp.linspace(-1.0, 1.0, 400)

    axes[0].plot(x_plot, u1(x_plot), 'b', linewidth=1.8, label="chebfunjax")
    axes[0].plot(x_plot, exact_shock(np.array(x_plot), eps), 'r--',
                 linewidth=1.2, label="exact erf")
    axes[0].set_title(f"Viscous shock (ε={eps})", fontsize=10)
    axes[0].legend(fontsize=7)

    axes[1].plot(x_plot, u2(x_plot), 'b', linewidth=1.8)
    axes[1].set_title(f"Interior layer at x=0.3 (ε={eps2})", fontsize=10)
    axes[1].axvline(0.3, color='k', linestyle='--', linewidth=0.8)

    axes[2].plot(x_plot, u3(x_plot), 'b', linewidth=1.8, label="chebfunjax")
    axes[2].plot(x_test3, exact3(x_test3), 'r--', linewidth=1.2, label="Airy")
    axes[2].set_title(f"Airy equation (ε={eps3})", fontsize=10)
    axes[2].legend(fontsize=7)

    fig.suptitle("Lee & Greengard stiff ODE examples", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "lee_greengard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
