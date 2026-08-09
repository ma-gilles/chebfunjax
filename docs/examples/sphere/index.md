# Sphere Examples (Spherefun)

Spherefun represents functions on the unit sphere `S²` using a double Fourier
series in spherical coordinates `(λ, θ)` (longitude, colatitude).

| Example | Description |
|---------|-------------|

| [Spherical harmonics (replica)](SphericalHarmonics.md) | Faithful replica: eigen-identity exactly 0; projection error to 14 digits. |
| [Rotating functions on the sphere (replica)](SpherefunRotate.md) | Faithful replica: ranks 29/74/139 vs published 29/74/141; shell-exact harmonics. |
| [Heat equation on the unit sphere (replica)](SphereHeatConduction.md) | Faithful replica: BDF2 error to 10 digits; mean conserved exactly. |
| [Parity partitioning a spherefun (replica)](SpherefunPartition.md) | Faithful replica: rank split 21 = 11 + 10 exact; sums identical. |
| [Advection-diffusion in the unit ball (replica)](AdvectionDiffusion.md) | Faithful replica: 150 IMEX steps, panel-for-panel spiral winding. |
| [The Laplace equation on the unit ball (replica)](LaplaceBall.md) | Honest partial: inner-mean identity exact; helmholtz mode defect ledgered. |
| [Helmholtz-Hodge decomposition (replica)](HelmholtzDecomposition.md) | Faithful replica: decomposition residual 5.3e-13; DFS spectral calculus. |
| [Solid harmonics (replica)](SolidHarmonics.md) | Faithful replica: harmonic to 4e-14, orthonormal to 1e-16. |
| [Poloidal-toroidal decomposition (replica)](PTDecomposition.md) | Faithful replica: div 1.3e-10, round-trip 1.2e-12. |
| [Helmholtz decomposition in the ball (replica)](HelmholtzDecompositionBall.md) | Faithful replica: all four identity norms in class or better. |
| [Gravitational force from a spherical shell (replica)](Gravity.md) | Faithful replica: Newton's theorem force to all 15 published digits. |
| [The Rayleigh quotient on the sphere (replica)](RayleighQuotientExample.md) | Faithful replica: all three eigenvalue errors at machine precision. |
| [Atmospheric temperature data (replica)](AtmosphericTemperature.md) | Faithful replica: pole values to 13 digits from the real dataset. |
