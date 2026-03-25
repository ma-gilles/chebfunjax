"""Double-well Schrodinger eigenstates.

The Schrodinger equation with a smooth double-well potential:
    -c * u''(x) + V(x)*u(x) = lambda * u(x), u(-1) = u(1) = 0

where V(x) = 5*(x^2 - 0.5)^2 has minima at x = ±1/sqrt(2) and a
barrier at x=0, creating a quantum double well.

Credit: Inspired by Chebfun example ode-eig/DoubleWell.m.
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Double-well Schrodinger equation eigenstates")
    print("=" * 60)

    c = 0.007
    dom = (-1.0, 1.0)
    n_eigs = 6

    # Use a smooth double-well potential: V(x) = 5*(x^2 - 0.5)^2
    # This has minima at x = ±1/sqrt(2) ≈ ±0.707 and a barrier at x=0
    N = Chebop(
        lambda x, u: -c * u.diff(2) + 5.0 * (x**2 - 0.5)**2 * u,
        domain=dom
    )
    N.lbc = 0.0
    N.rbc = 0.0

    lam = N.eigs(k=n_eigs)
    lam_real = np.sort(np.real(np.array(lam)))

    print(f"\nDouble-well potential V(x)=5*(x^2-0.5)^2, c={c}:")
    print(f"  First {n_eigs} eigenvalues:")
    for i, l in enumerate(lam_real):
        print(f"    lam_{i+1} = {l:.8f}")

    # Basic sanity checks:
    # All eigenvalues should be positive (V >= 0 and c > 0)
    assert np.all(lam_real > 0), "Eigenvalues should be positive"
    # They should be increasing
    assert np.all(np.diff(lam_real) > 0), "Eigenvalues should be increasing"
    # First eigenvalue is the ground state energy
    print(f"\n  Ground state energy: lam_1 = {lam_real[0]:.6f}")

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
