# ODE Eigenvalue Examples

Differential operators have spectra — eigenvalues and eigenfunctions.
Chebfun computes these spectra via the `eigs` method.

| Example | Description |
|---------|-------------|
| [Continuous analogue of Wilkinson matrix (replica)](ContinuousWilkinson.md) | Faithful replica: near-degenerate pairs to 12 digits; pseudo-eigenfunction residual matches to 4 digits. |
| [Eigenvalues by contour integral projection (replica)](ContourProjEig.md) | Faithful replica: FEAST-like projection to 12-13 digits of the published values. |
| [Double-well Schroedinger eigenstates (replica)](DoubleWell.md) | Faithful replica: 12 eigenvalues to 11 digits via piecewise eigs. |
| [Frequencies of a drum (replica)](Drum.md) | Faithful replica: J0 zeros to 1e-10; octave design astar to 10 digits. |
| [Eigenstates of the Schroedinger equation (replica)](Eigenstates.md) | Faithful replica: nine potentials; harmonic-oscillator eigenvalues to 13 digits. |
| [Periodic ODE eigenvalue problems (replica)](FourierEigs.md) | Faithful replica: -u''=lam u and Mathieu characteristic values to 1e-13. |
| [Landscape function and localization of eigenfunctions (replica)](Landscape.md) | Faithful replica: same well-by-well localization; eigenvalues to 7 digits. |
| [Avoided crossings for ODE eigenvalues (replica)](LevelRepulsionODE.md) | Faithful replica: 4th-order clamped operator; smooth repelling curves. |
| [The nullspace of a linear operator (replica)](NullSpace.md) | Faithful replica: Chebop.null with exotic integral conditions; minE/bc_star to 10-11 digits. |
| [The nonlinear optical response of a simple molecule (replica)](OpticalResponse.md) | Faithful replica: alpha = -1/4 to 5e-11, beta = gamma = 0. |
| [Orr-Sommerfeld eigenvalues (replica)](OrrSommerfeld.md) | Faithful replica: lambda_r matches MATLAB R2025b at Re=2000 and the critical Re. |
| [Eigenvalues of random operators (replica)](Randfuneig.md) | Faithful replica: circular law + Fredholm eig(chebfun2) samples. |
| [Rayleigh quotient iteration for an operator (replica)](RayleighQuotient.md) | Faithful replica: MATLAB rng data inlined; iterates match digit-for-digit. |
| [Model of a quantum dot array for solar energy (replica)](SolarQDA.md) | Faithful replica: all eight energies to 10-11 digits; delocalization figures. |
| [Stability of a thermoelastic rod (replica)](ThermoelasticRod.md) | Faithful replica: Barber-condition eigenvalues to 10 digits; dstar = 1 to 1e-9. |
| [Wave equation with decay band (replica)](WaveDecay.md) | Faithful replica: modes 1, 2, 20, 40 with and without the decay band. |
