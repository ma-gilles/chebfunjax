# A parameter dependent ODE with breakpoints

*Asgeir Birkisson, January 2012*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/ode-linear/ParameterODE.html)

(Chebfun example ode-linear/ParameterODE.m)

Let the ODE boundary-value problem

$$ (a(x,s)u')' = 1, \qquad u(0) = u(1) = 0 $$

be given, where $a(x,s) = 1 + 4s(x^2 - x)$. The exact solution is

$$ u(x,s) = \frac{1}{8s}\log(1 + 4s(x^2-x)) = \frac{1}{8s}\log(a(x,s)). $$

For $s = 1$ the solution has a singularity at $x = 1/2$; here we explore
what happens as $s = 1 - 10^{-\gamma}$ approaches that critical value.
Rewriting as $a u'' + a' u' = 1$ and solving on the plain domain
$[0,1]$ for $\gamma = 1, 2, 3$:

![ParameterODE figure 1](../../images/ode-linear/ParameterODE_repl_01.png)

![ParameterODE figure 2](../../images/ode-linear/ParameterODE_repl_02.png)

![ParameterODE figure 3](../../images/ode-linear/ParameterODE_repl_03.png)

The residual and error grow steadily with $\gamma$ as the near-singular
coefficient gets harder to resolve:

![ParameterODE figure 4](../../images/ode-linear/ParameterODE_repl_04.png)

![ParameterODE figure 5](../../images/ode-linear/ParameterODE_repl_05.png)

## Introducing a breakpoint

Since we know the trouble concentrates at $x = 1/2$, we add a breakpoint
there — the piecewise discretization can then stack resolution exactly
where it is needed, and now $\gamma$ can go all the way to 7:

![ParameterODE figure 6](../../images/ode-linear/ParameterODE_repl_06.png)

![ParameterODE figure 7](../../images/ode-linear/ParameterODE_repl_07.png)

![ParameterODE figure 8](../../images/ode-linear/ParameterODE_repl_08.png)

![ParameterODE figure 9](../../images/ode-linear/ParameterODE_repl_09.png)

![ParameterODE figure 10](../../images/ode-linear/ParameterODE_repl_10.png)

![ParameterODE figure 11](../../images/ode-linear/ParameterODE_repl_11.png)

![ParameterODE figure 12](../../images/ode-linear/ParameterODE_repl_12.png)

The error with breakpoints stays at the 1e-10-1e-12 level through
$\gamma = 6$ ($3.3\times 10^{-13}$ at $\gamma = 1$), rising only at
$\gamma = 7$ where the per-piece resolution cap is reached:

![ParameterODE figure 13](../../images/ode-linear/ParameterODE_repl_13.png)

(The published page's figure-title lengths are MATLAB's adaptive
dimensions; chebfunjax's adaptive lengths differ in detail but tell the
same story — the breakpoint restores solvability far beyond what the
plain domain can reach.)

## References

1. P. G. Constantine, personal communication.

---

*Replica script: [`examples/ode-linear/parameter_ode_replica.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/ode-linear/parameter_ode_replica.py).
Original example copyright by The University of Oxford and The Chebfun
Developers.*
