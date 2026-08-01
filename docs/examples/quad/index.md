# Quadrature Examples

Chebfun integrates smooth functions by analytically integrating the
Chebyshev expansion. These examples explore Gauss, Clenshaw-Curtis,
and related quadrature rules.

| Example | Description |
|---------|-------------|
| [Gauss and Clenshaw-Curtis quadrature (replica)](GaussClenCurt.md) | Faithful replica: sum, Clenshaw-Curtis, and Gauss quadrature agree for a wiggly function; convergence curves compared. |
| [Hermite quadrature (replica)](HermiteQuad.md) | Faithful replica: Gauss-Hermite quadrature converges super-exponentially; every table entry matches digit-for-digit. |
| [Spike integral (replica)](SpikeIntegral.md) | Faithful replica: integrating the Kahaner spike function with splitting on. |
| [Quadrature convergence rates (replica)](QuadratureConvergence.md) | Faithful replica: convergence-rate comparison of Gauss and Clenshaw-Curtis on functions of varying smoothness. |
| [Symbolic and numeric integration (replica)](SymbolicNumeric.md) | Faithful replica: chebfun integrals versus published closed forms. |
| [Sumdisk for integration over a disk (replica)](SumdiskDemo.md) | Faithful replica: chebfun2 sumdisk integrates over the inscribed disk via closed-form Chebyshev-product disk integrals. |
| [Integrating Tj(x)Tk(y) over the unit disk (replica)](TjTkDisk.md) | Faithful replica: disk integrals of Chebyshev products vanish unless j,k are even and differ by 0 or 2; the tridiagonal matrix reproduces sign-for-sign. |
| [Battery test of Chebfun as a general-purpose integrator](battery_test.md) | This example applies chebfunjax to a selection of the Kahaner battery of test integrands — functions that are challen... |
| [Hermite quadrature](hermite_quad.md) | Gauss–Hermite quadrature integrates functions of the form ... over ... using ... carefully chosen nodes and weights: |
| [Spike integral](spike_integral.md) | The spike function (also known as F21F in the Kahaner benchmark) consists of several spikes of increasing sharpness: |
| [Symbolic and numerical integration](symbolic_numeric.md) | Computer algebra systems can compute remarkable symbolic integrals. Here we compare several such results with direct ... |
| [Clenshaw–Curtis Quadrature](clenshaw_curtis.md) | Clenshaw–Curtis quadrature uses the values of ... at Chebyshev nodes ... to form an exact integral of the degree-... ... |
| [Convergence Rates for Quadrature](convergence_rates.md) | The convergence rate of a quadrature rule depends on the smoothness of the integrand. For analytic functions, Gauss a... |
| [Gauss Quadrature](gauss_quadrature.md) | Gauss–Legendre quadrature uses ... carefully chosen nodes and weights to integrate polynomials of degree up to ... ex... |
| [Gauss and Clenshaw-Curtis Quadrature](gauss_vs_clenshaw_curtis.md) | Two of the most important quadrature rules for smooth functions are Gauss-Legendre and Clenshaw-Curtis quadrature. Bo... |
| [Tricky Integrals](tricky_integrals.md) | Some integrals are challenging for standard quadrature because of near-singularities, oscillatory integrands, or endp... |
