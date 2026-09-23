# ODE Eigenvalue Examples

Differential operators have spectra — eigenvalues and eigenfunctions.
Chebfun computes these spectra via the `eigs` method.

| Example | Description |
|---------|-------------|
| [Continuous analogue of Wilkinson matrix](ContinuousWilkinson.md) | translation: near-degenerate pairs to 12 digits; pseudo-eigenfunction residual matches to 4 digits. |
| [Eigenvalues by contour integral projection](ContourProjEig.md) | translation: FEAST-like projection to 12-13 digits of the published values. |
| [Double-well Schroedinger eigenstates](DoubleWell.md) | translation: 12 eigenvalues to 11 digits via piecewise eigs. |
| [Frequencies of a drum](Drum.md) | translation: J0 zeros to 1e-10; octave design astar to 10 digits. |
| [Eigenstates of the Schroedinger equation](Eigenstates.md) | translation: nine potentials; harmonic-oscillator eigenvalues to 13 digits. |
| [Periodic ODE eigenvalue problems](FourierEigs.md) | translation: -u''=lam u and Mathieu characteristic values to 1e-13. |
| [Landscape function and localization of eigenfunctions](Landscape.md) | translation: same well-by-well localization; eigenvalues to 7 digits. |
| [Avoided crossings for ODE eigenvalues](LevelRepulsionODE.md) | translation: 4th-order clamped operator; smooth repelling curves. |
| [The nullspace of a linear operator](NullSpace.md) | translation: Chebop.null with exotic integral conditions; minE/bc_star to 10-11 digits. |
| [The nonlinear optical response of a simple molecule](OpticalResponse.md) | translation: alpha = -1/4 to 5e-11, beta = gamma = 0. |
| [Orr-Sommerfeld eigenvalues](OrrSommerfeld.md) | translation: lambda_r matches MATLAB R2025b at Re=2000 and the critical Re. |
| [Eigenvalues of random operators](Randfuneig.md) | translation: circular law + Fredholm eig(chebfun2) samples. |
| [Rayleigh quotient iteration for an operator](RayleighQuotient.md) | translation: MATLAB rng data inlined; iterates match digit-for-digit. |
| [Model of a quantum dot array for solar energy](SolarQDA.md) | translation: all eight energies to 10-11 digits; delocalization figures. |
| [Stability of a thermoelastic rod](ThermoelasticRod.md) | translation: Barber-condition eigenvalues to 10 digits; dstar = 1 to 1e-9. |
| [Wave equation with decay band](WaveDecay.md) | translation: modes 1, 2, 20, 40 with and without the decay band. |
