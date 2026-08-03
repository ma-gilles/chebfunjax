# Optimization Examples

Chebfun computes global minima and maxima of smooth functions via rootfinding
on the derivative — no gradient descent needed.

| Example | Description |
|---------|-------------|
| [Six-hump camel function (replica)](DixonSzego.md) | Faithful replica: global minimum to 15 digits via chebfun2 min2. |
| [Optimization of the Rosenbrock function (replica)](Rosenbrock.md) | Faithful replica: nested 1D minimization; minima digit-for-digit. |
| [SIAM 100-digit challenge minimum (replica)](GlobalMinimum.md) | Faithful replica: rank 4, minimum to 4.4e-15 (beats published 4.5e-13). |
| [Optimization over an integral (replica)](OptimInt.md) | Faithful replica: parametrized-integral chebfun; roots to 13 digits, max digit-for-digit. |
| [The catenary by variational Newton (replica)](Catenary.md) | Faithful replica: chebop accessory-equation Newton — J sequence to 12-13 digits. |
| [Constrained extrema via composition (replica)](ConstrainedExtrema.md) | Faithful replica: constrained optima digit-for-digit, no Lagrange multipliers. |
