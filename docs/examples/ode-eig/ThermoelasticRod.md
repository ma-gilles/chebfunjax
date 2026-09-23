# Stability of a thermoelastic rod

*Toby Driscoll, November 2011*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/ode-eig/ThermoelasticRod.html)

Python translation: [`examples/ode-eig/thermoelasticrod.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/ode-eig/thermoelasticrod.py)

Suppose a thermoelastic rod is fixed to a wall at one end and may expand to make contact with a wall at the other end. J. R. Barber [1] proposed a boundary condition that models a physically realistic transition between thermal insulation, when far from contact, and perfect thermal contact.

Linear stability analysis suggests a change from stable to unstable behavior as the temperature difference between the walls increases. The eigenvalue problem governing the stability of the perturbation $\phi(x)$ is nondimensionally

$$ \phi''(x) = \lambda \phi(x),\qquad 0 < x < 1, $$

$$ \phi(0) = 0,~~ \phi'(1) + \phi(1) = 4 \delta\int_0^1\phi(x) dx , $$

where the value of $\delta$ is a function of the thermal gradient. The transition from stable to unstable happens at $\delta=1$. The presence of the integral of $\phi$ in the boundary condition makes the problem unusual from a classical standpoint, but from the Chebfun point of view it's just another linear boundary condition.

First, we solve the eigenvalue problem in a stable case.

```matlab
N = chebop( @(x,u) diff(u,2), [0 1] );    % operator on 0<x<1
N.lbc = 0;              % fixed end
delta = 0.96;           % stable choice
N.bc = @(x,u) feval(diff(u),1) + u(1) - 4*delta*sum(u);  % Barber condition
[Vs,Ls] = eigs(N,4,0);  % eigenmodes closest to zero
```

The eigenvalues are all negative, indicating stability:

```matlab
format long
diag(Ls)
```

```text
ans =
   1.0e+02 *
  -1.234915472724549
  -0.626486098335068
  -0.251462532662628
  -0.001601435706437
```

Here is what happens in a slightly unstable case:

```matlab
delta = 1.02;  % unstable choice
N.bc = @(x,u) feval(diff(u),1) + u(1) - 4*delta*sum(u);  % Barber condition
[Vu,Lu] = eigs(N,4,0);
diag(Lu)
```

```text
ans =
   1.0e+02 *
  -1.235278901225335
  -0.625884455972551
  -0.252000055361275
   0.000799646107565
```

Here we see the perturbation which is least stable in the first case, or unstable in the second case.

```matlab
LW = 'linewidth'; MS = 'markersize';
subplot(1,2,1)
plot(Vs(:,4),LW,1.6)
title(sprintf('Stable, \\lambda = %.3f',Ls(4,4)))
subplot(1,2,2)
plot(Vu(:,4),LW,1.6)
title(sprintf('Unstable, \\lambda = %.3f',Lu(4,4)))
```

![ThermoelasticRod figure 01](../../images/ode-eig/ThermoelasticRod_01.png)

The solutions above look linear, but they do have significant Chebyshev coefficients out to degree 8.

Without knowing the transition value $\delta=1$ in advance, we could locate it through a simple Chebfun rootfinding search. First, we parameterize the boundary conditions and the maximum real eigenvalue.

```matlab
BC = @(delta) @(x,u) [u(0); feval(diff(u),1) + u(1) - 4*delta*sum(u)];
maxlam = @(delta) eigs( chebop(@(x,u)diff(u,2),[0 1],BC(delta)), 1, 0 );
```

Then, we construct a chebfun for the maximum $\lambda$. A polynomial of degree 10 captures the behavior of the maximum eigenvalue to about 11 digits.

```matlab
stability = chebfun(maxlam,[0.5,2],'eps',1e-11,'vectorize')
```

```text
stability =
   chebfun column (1 smooth piece)
       interval       length     endpoint values
[     0.5,       2]       11        -2      3.9
vertical scale = 3.9
```

Finally, the transition in stability occurs when the eigenvalue passes through zero.

```matlab
dstar = find(stability==0)
clf, plot(stability,LW,1.6), hold on, plot(dstar,0,'ro',MS,16)
xlabel('\delta'), ylabel('max \lambda'), grid on
```

```text
dstar =
   1.000000000002016
```

![ThermoelasticRod figure 02](../../images/ode-eig/ThermoelasticRod_02.png)

## References

1. J. R. Barber, "Contact problems involving a cooled punch," *Journal of Elasticity*, 8 (1978), 409-423.
2. J. A. Pelesko, "Nonlinear stability, thermoelastic contact, and the Barber condition", *Journal of Applied Mechanics*, 68 (2001), 28-33.

---

*Translated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); prose and MATLAB code from the original example, copyright The University of Oxford and The Chebfun Developers.  Printed outputs and figures are chebfunjax's.*
