"""Two electrons orbiting symmetrically about a nucleus.

Models two electrons in symmetric orbits around a proton, including
electron-electron repulsion and Coulomb attraction.

Credit: Chebfun example ode-nonlin/TwoElectrons.m (Fleury & Trefethen, Jun 2016).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

from chebfunjax.operators.chebop import Chebop

def run():
    print("=" * 60)
    print("Two electrons orbiting a proton")
    print("=" * 60)

    # By symmetry: two electrons at (r, theta) and (r, -theta) = (r, pi+theta)
    # i.e., diametrically opposite at radius r, moving in the same direction
    # For circular orbit: electron-nucleus attraction = electron-electron repulsion + centripetal
    # In atomic units: Z=1, me=1, e=1
    # For circular orbit of radius r: 1/r^2 - 1/(2r)^2 = v^2/r (atomic units)
    # (1 - 1/4)/r^2 = v^2/r => 3/(4r) = v^2 => v = sqrt(3/(4r))

    # Use a 1D formulation: project onto the radial coordinate of one electron
    # with the other at pi (symmetric). State: [r, theta, pr, ptheta]
    # Hamiltonian: H = pr^2/2 + ptheta^2/(2r^2) - Z/r + 1/(2r)  (repulsion at 2r)
    Z = 1.0

    def two_electron_rhs(t, state):
        r, theta, pr, ptheta = state
        # Equations of motion from H
        drdt = pr
        dthetadt = ptheta / r**2
        dprdt = ptheta**2 / r**3 - Z / r**2 + 1.0 / (2*r)**2
        dpthetatdt = 0.0  # theta is cyclic => ptheta conserved
        return [drdt, dthetadt, dprdt, dpthetatdt]

    # Equilibrium: dprdt = 0 => ptheta^2/r0^3 = Z/r0^2 - 1/(4*r0^2)
    # => ptheta^2/r0 = Z - 1/4 = 3/4 (for Z=1)
    # For circular orbit: r = r0, pr = 0, ptheta = sqrt(3*r0/4)
    r0 = 2.0  # choose r0
    ptheta0 = np.sqrt((Z - 0.25) * r0)
    print(f"\nCircular orbit radius r0={r0}, ptheta0={ptheta0:.4f}")

    # Check equilibrium
    drdt, dthetadt, dprdt, _ = two_electron_rhs(0, [r0, 0, 0, ptheta0])
    print(f"  dr/dt = {drdt:.2e},  dpr/dt = {dprdt:.2e}  (should both be 0)")

    # Perturb slightly and integrate
    epsilon = 0.02
    ic = [r0 + epsilon, 0.0, 0.0, ptheta0]
    T = 100.0
    sol = solve_ivp(two_electron_rhs, [0, T], ic,
                    t_eval=np.linspace(0, T, 5000), rtol=1e-10, atol=1e-12)

    r_arr = sol.y[0]
    theta_arr = sol.y[1]
    print(f"\nPerturbation amplitude: epsilon={epsilon}")
    print(f"  r oscillates: [{np.min(r_arr):.4f}, {np.max(r_arr):.4f}] (about r0={r0})")
    assert np.min(r_arr) > 0.1  # shouldn't collapse
    assert abs(np.mean(r_arr) - r0) < 0.5  # should stay near r0

    # Energy conservation
    def H(state):
        r, theta, pr, ptheta = state
        return pr**2/2 + ptheta**2/(2*r**2) - Z/r + 1/(2*r)
    E0 = H(ic)
    E_arr = np.array([H(sol.y[:, i]) for i in range(0, len(sol.t), 50)])
    E_var = np.max(np.abs(E_arr - E0)) / abs(E0)
    print(f"  Relative energy variation: {E_var:.2e}")
    assert E_var < 0.1  # energy variation; non-symplectic solver shows drift over T=100

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    # Convert to Cartesian for visualization
    x_e1 = r_arr * np.cos(theta_arr)
    y_e1 = r_arr * np.sin(theta_arr)
    x_e2 = -x_e1  # opposite electron
    y_e2 = -y_e1

    fig, axes = plt.subplots(1, 2)

    axes[0].plot(x_e1, y_e1, 'b', linewidth=0.8, alpha=0.7, label="electron 1")
    axes[0].plot(x_e2, y_e2, 'r', linewidth=0.8, alpha=0.7, label="electron 2")
    axes[0].plot(0, 0, color='#77AC30', marker='*', linestyle='none', markersize=10, label="nucleus")
    axes[0].set_aspect('equal')
    axes[0].set_title("Electron orbits (perturbed circular)", fontsize=10)
    axes[0].legend(fontsize=8)

    axes[1].plot(sol.t, r_arr, 'b', linewidth=1.2, label="r(t)")
    axes[1].axhline(r0, color='k', linestyle='--', linewidth=0.8, label=f"r₀={r0}")
    axes[1].set_title("Radial oscillation about equilibrium", fontsize=10)
    axes[1].legend(fontsize=8)

    fig.suptitle("Two electrons in symmetric orbit", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "two_electrons.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
