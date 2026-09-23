# Sphere Examples (Spherefun)

Spherefun represents functions on the unit sphere `S²` using a double Fourier
series in spherical coordinates `(λ, θ)` (longitude, colatitude).

| Example | Description |
|---------|-------------|

| [Spherical harmonics](SphericalHarmonics.md) | translation: eigen-identity exactly 0; projection error to 14 digits. |
| [Rotating functions on the sphere](SpherefunRotate.md) | translation: ranks 29/74/139 vs published 29/74/141; shell-exact harmonics. |
| [Heat equation on the unit sphere](SphereHeatConduction.md) | translation: BDF2 error to 10 digits; mean conserved exactly. |
| [Parity partitioning a spherefun](SpherefunPartition.md) | translation: rank split 21 = 11 + 10 exact; sums identical. |
| [Advection-diffusion in the unit ball](AdvectionDiffusion.md) | translation: 150 IMEX steps, panel-for-panel spiral winding. |
| [The Laplace equation on the unit ball](LaplaceBall.md) | Honest partial: inner-mean identity exact; helmholtz mode defect ledgered. |
| [Helmholtz-Hodge decomposition](HelmholtzDecomposition.md) | translation: decomposition residual 5.3e-13; DFS spectral calculus. |
| [Solid harmonics](SolidHarmonics.md) | translation: harmonic to 4e-14, orthonormal to 1e-16. |
| [Poloidal-toroidal decomposition](PTDecomposition.md) | translation: div 1.3e-10, round-trip 1.2e-12. |
| [Helmholtz decomposition in the ball](HelmholtzDecompositionBall.md) | translation: all four identity norms in class or better. |
| [Gravitational force from a spherical shell](Gravity.md) | translation: Newton's theorem force to all 15 published digits. |
| [The Rayleigh quotient on the sphere](RayleighQuotientExample.md) | translation: all three eigenvalue errors at machine precision. |
| [Atmospheric temperature data](AtmosphericTemperature.md) | translation: pole values to 13 digits from the real dataset. |
