# Optimization Examples

Chebfun computes global minima and maxima of smooth functions via rootfinding
on the derivative — no gradient descent needed.

| Example | Description |
|---------|-------------|
| [Six-hump camel function](DixonSzego.md) | translation: global minimum to 15 digits via chebfun2 min2. |
| [Optimization of the Rosenbrock function](Rosenbrock.md) | translation: nested 1D minimization; minima digit-for-digit. |
| [SIAM 100-digit challenge minimum](GlobalMinimum.md) | translation: rank 4, minimum to 4.4e-15 (beats published 4.5e-13). |
| [Optimization over an integral](OptimInt.md) | translation: parametrized-integral chebfun; roots to 13 digits, max digit-for-digit. |
| [The catenary by variational Newton](Catenary.md) | translation: chebop accessory-equation Newton — J sequence to 12-13 digits. |
| [Constrained extrema via composition](ConstrainedExtrema.md) | translation: constrained optima digit-for-digit, no Lagrange multipliers. |
| [Closest approach of Mercury and Earth](MercuryEarth.md) | translation: global minimum of the distance chebfun. |
| [Rosenbrock revisited with chebfun2](Rosenbrock2.md) | translation: one-call min2 + gradient critical points. |
| [Constrained optimization](ConstrainedOptimization.md) | translation: indicator constraints + heart-region maximum to 15 digits. |
| [The lowest position of a resting needle](Needle.md) | translation: nonsmooth resting-height landscape + simplex polish. |
| [Extrema of complicated functions](ExtremeExtrema.md) | translation: global max through abs/min nonsmoothness, 13-15 digits — OPT CATEGORY COMPLETE. |
