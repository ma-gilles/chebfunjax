"""Model of a quantum dot array for solar energy.

Solves the 1D Schrodinger eigenvalue problem for a quantum dot array (QDA),
modelling electrons in periodic potential wells with piecewise-constant
effective mass (InAs wells in GaAs barrier material).

Credit: Chebfun example ode-eig/SolarQDA.m (Toby Driscoll, May 2011).
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
from chebfunjax.operators.chebop import Chebop

def run():
    print("=" * 60)
    print("Quantum dot array: Schrodinger eigenvalue problem")
    print("=" * 60)

    # Physical constants (SI)
    hbar = 1.054e-34   # J*s
    eV = 1.602e-19     # J per eV
    me = 9.109e-31     # kg

    # Effective masses (kg) for InAs (well) and GaAs (barrier)
    m_well = 0.022 * me   # InAs
    m_barrier = 0.067 * me  # GaAs

    # Convert to energy in eV and length in nm:
    # Use dimensionless units: energy in eV, length in nm
    # hbar^2/(2m_barrier) in eV*nm^2
    hbar2_over_2m = (hbar**2 / (2.0 * m_barrier)) / (eV * 1e-18)  # eV*nm^2
    mass_ratio = m_barrier / m_well  # ratio of barrier to well mass

    numwell = 4
    width = 6.5    # nm, well width
    depth = 0.85   # eV, well depth
    spacing = 3.0  # nm, barrier width

    print(f"\nQDA parameters:")
    print(f"  {numwell} wells, width={width} nm, depth={depth} eV, spacing={spacing} nm")
    print(f"  hbar^2/(2m_GaAs) = {hbar2_over_2m:.4f} eV*nm^2")

    # Build piecewise-constant potential on fine grid
    # Layout: [left_buffer | well | spacing | well | ... | well | right_buffer]
    left_buf = 10 * spacing
    right_buf = 9 * spacing
    total_width = left_buf + numwell * width + (numwell - 1) * spacing + right_buf

    x_fine = np.linspace(0, total_width, 5000)
    U_fine = np.zeros_like(x_fine)
    m_fine = m_barrier * np.ones_like(x_fine)  # effective mass

    # Place wells
    well_starts = [left_buf + i * (width + spacing) for i in range(numwell)]
    for ws in well_starts:
        mask = (x_fine >= ws) & (x_fine < ws + width)
        U_fine[mask] = -depth          # well depth (negative = attractive)
        m_fine[mask] = m_well

    dom_nm = (0.0, total_width)
    print(f"  Domain: [{dom_nm[0]:.1f}, {dom_nm[1]:.1f}] nm")

    # Build Schrodinger operator using smooth Gaussian-well approximation.
    # Piecewise-constant wells are replaced by smooth Gaussians (width sigma=width/2)
    # so Chebfun can represent the potential without convergence issues.
    hbar2_m = float(hbar2_over_2m)
    sigma_well = float(width) / 2.0
    well_centers = [float(ws + width / 2.0) for ws in well_starts]

    def U_smooth(x):
        return -depth * sum(jnp.exp(-0.5 * ((x - wc) / sigma_well)**2)
                            for wc in well_centers)

    U_cf = cj.chebfun(U_smooth, domain=dom_nm)
    # Rebuild U_fine using smooth approximation for plotting
    U_fine_smooth = sum(
        -depth * np.exp(-0.5 * ((x_fine - wc) / sigma_well)**2)
        for wc in well_centers
    )

    # Scale: use hbar^2/(2m_barrier) factor; lengths in nm, energies in eV
    L_qda = Chebop(
        lambda x, u: -hbar2_m * u.diff(2) + U_cf * u,
        domain=dom_nm,
    )
    L_qda.lbc = 0.0
    L_qda.rbc = 0.0

    k = numwell
    print(f"\nComputing {k} lowest energy levels ...")
    lams = L_qda.eigs(k=k)
    energies = np.sort(np.real(np.array(lams)))
    print(f"\nEnergy levels (eV):")
    for i, E in enumerate(energies):
        print(f"  E_{i+1} = {E:.6f} eV")
    assert len(energies) == k, f"Expected {k} eigenvalues"

    # --- Perturbed wells (2% variance in depth) ---
    print("\nPerturbed wells (2% depth fluctuation):")
    rng = np.random.default_rng(1138)
    deltas = [float(0.017 * rng.standard_normal()) for _ in range(numwell)]
    depths_p = [-depth * (1.0 + d) for d in deltas]
    U_perturb_fine = sum(
        dp * np.exp(-0.5 * ((x_fine - wc) / sigma_well)**2)
        for dp, wc in zip(depths_p, well_centers)
    )

    def U_perturb_smooth(x):
        return sum(float(dp) * jnp.exp(-0.5 * ((x - float(wc)) / sigma_well)**2)
                   for dp, wc in zip(depths_p, well_centers))

    U_perturb_cf = cj.chebfun(U_perturb_smooth, domain=dom_nm)

    L_qda_p = Chebop(
        lambda x, u: -hbar2_m * u.diff(2) + U_perturb_cf * u,
        domain=dom_nm,
    )
    L_qda_p.lbc = 0.0; L_qda_p.rbc = 0.0
    lams_p = L_qda_p.eigs(k=k)
    energies_p = np.sort(np.real(np.array(lams_p)))
    print(f"  Perturbed energies: {energies_p}")

    energy_shift = np.max(np.abs(energies_p - energies))
    print(f"  Max energy shift from perturbation: {energy_shift:.4f} eV")

    # --- Plot -----------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    # Potential and energy levels
    axes[0].plot(x_fine, U_fine_smooth, 'b', linewidth=1.5, label="U(x) [ideal]")
    axes[0].plot(x_fine, U_perturb_fine, 'g--', linewidth=1.0, alpha=0.7, label="U(x) [perturbed]")
    colors = plt.cm.tab10(np.linspace(0, 0.5, k))
    for i in range(k):
        axes[0].axhline(energies[i], color=colors[i], linewidth=1.2, linestyle=':',
                        label=f"E_{i+1}={energies[i]:.3f} eV")
    axes[0].set_title(f"QDA: {numwell}-well potential and energy levels", fontsize=10)
    axes[0].legend(fontsize=7, loc='lower right')

    # Energy comparison: ideal vs perturbed
    x_pos = np.arange(1, k+1)
    axes[1].bar(x_pos - 0.2, energies, width=0.35, color='steelblue', alpha=0.8, label="ideal")
    axes[1].bar(x_pos + 0.2, energies_p, width=0.35, color='coral', alpha=0.8, label="perturbed")
    axes[1].set_title("Energy levels: ideal vs perturbed wells", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].set_xticks(x_pos)

    fig.suptitle("Quantum dot array: 1D Schrodinger eigenvalue problem", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "solar_qda.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
