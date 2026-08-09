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
| [Some tricky integrals (replica)](Tricky.md) | Faithful replica: ten challenge integrals; all reproduce, and the 2979-discontinuity case comes out ten digits more accurate than the published MATLAB run. |
