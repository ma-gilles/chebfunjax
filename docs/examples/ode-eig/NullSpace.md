# The nullspace of a linear operator

*Nick Hale and Stefan Guettel, December 2011*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/ode-eig/NullSpace.html)

Python translation: [`examples/ode-eig/nullspace.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/ode-eig/nullspace.py)

We've recently introduced some new functionality in Chebfun for computing the nullspace of differential operators. Let's explore this with a couple of simple examples.

## 1. Simple example #1

Let's start as simply as we can, and take

$$ (Lu)(x) := u''(x), \quad x\in [-1, 1]. $$

```matlab
L = chebop(@(u) diff(u, 2));
```

Clearly the nullspace of this operator -- that is, the space of functions $v$ for which $L(v)=0$ -- is spanned the two functions

```matlab
v = [1, chebfun('x')];
norm(L(v))
```

```text
ans =
     0
```

Supposing we didn't know this, we could compute a basis for the nullspace with the `null` method:

```matlab
LW = 'LineWidth'; lw = 1.6;
V = null(L)
plot(V, LW, lw)
V'*V
norm(L(V))
```

```text
V =
   chebfun column1 (1 smooth piece)
       interval       length     endpoint values
[      -1,       1]        2       1.3     -1.1
vertical scale = 1.3
   chebfun column2 (1 smooth piece)
       interval       length     endpoint values
[      -1,       1]        3      0.57     0.84
vertical scale = 0.84
```

![NullSpace figure 01](../../images/ode-eig/NullSpace_01.png)

where we find that $V^T V = I$ and $LV \approx 0$ as required.

Clearly `V` doesn't correspond directly to $1$ and $x$, since there is some freedom in how we orthogonalise the basis. However, we can check that `V` and ${1, x}$ correspond to the same spaces by computing the angle between the spaces with the `subspace` command.

```matlab
subspace(v, V)
```

```text
ans =
   1.000000000000000   0.000000000000000
   0.000000000000000   1.000000000000000
ans =
     1.683151518751667e-13
ans =
     2.494695367373348e-14
```

## 2. Incomplete boundary conditions

Now let's consider the more complicated 2nd-order operator

$$ Lu = u'' + 0.1x(1-x^2)u' + \sin(x)u, \quad x\in [-\pi, \pi]. \qquad (*) $$

```matlab
dom = [-pi, pi];
L = chebop(@(x, u) diff(u, 2) + .1*x.*(1-x.^2).*diff(u) + sin(x).*u, dom);
```

As before, it has a nullspace of rank 2.

```matlab
V = null(L)
plot(V, LW, lw)
V'*V
norm(L(V))
```

```text
V =
   chebfun column1 (1 smooth piece)
       interval       length     endpoint values
[    -3.1,     3.1]       41    -0.044      1.3
vertical scale = 1.3
   chebfun column2 (1 smooth piece)
       interval       length     endpoint values
[    -3.1,     3.1]       42       1.8    0.067
vertical scale = 1.8
ans =
   1.000000000000000  -0.000000000000000
  -0.000000000000000   1.000000000000000
ans =
     1.437867407068610e-10
```

![NullSpace figure 02](../../images/ode-eig/NullSpace_02.png)

However, now suppose we impose one boundary condition, say, a Dirichlet condition at the left. This removes one degree of freedom, and we are left with a rank 1 nullspace.

```matlab
L.lbc = 0;
L.rbc = [];
v = null(L)
plot(v, LW, lw), shg
v'*v
norm(L(v))
```

```text
v =
   chebfun column (1 smooth piece)
       interval       length     endpoint values
[    -3.1,     3.1]       41   1.4e-17      1.3
vertical scale = 1.3
```

![NullSpace figure 03](../../images/ode-eig/NullSpace_03.png)

Clearly this null vector must satisfy the given condition $v(-\pi) = 0.$

```matlab
v(-pi)
```

```text
ans =
   1.000000000000000
ans =
     1.320931260704616e-10
ans =
     0
```

## 3. An application

Where might these ideas be useful? Well, suppose we were interested in equation $(*)$ with a homogeneous Dirichlet condition at the left, and wanted to know what inhomogeneous Dirichlet condition gave the minimal 2-norm of the solution to $Lu = 1$. Rather than solving the linear system for a number of different boundary conditions (which would be computationally expensive) we could simply solve for one, say again a homogeneous Dirichlet condition,

```matlab
L.rbc = 0;
u = L\1;
hold on, plot(u, '--r', LW, lw), hold off
```

![NullSpace figure 04](../../images/ode-eig/NullSpace_04.png)

and compute the rest by adding a scalar multiple of the null-function $v$.

```matlab
E = chebfun(@(c) norm(u + c*v, 2), [-10, 10], 'vectorize', 'splitting', 'on');
plot(E,LW,lw)
```

![NullSpace figure 05](../../images/ode-eig/NullSpace_05.png)

We compute the 2-norm as a chebfun in the unknown variable $c$, which we can then minimise to obtain the minimal energy solution

```matlab
[minE, c_star] = min(E)
u_star = u + c_star*v
plot(u_star,LW,lw)
```

```text
minE =
   4.121950420615068
c_star =
   3.143771420955451
u_star =
   chebfun column (1 smooth piece)
       interval       length     endpoint values
[    -3.1,     3.1]       42     1e-14        4
vertical scale =   4
```

![NullSpace figure 06](../../images/ode-eig/NullSpace_06.png)

So the condition we require is that $u(\pi)$ = `bc_star`, where

```matlab
bc_star = u_star(pi)
```

```text
bc_star =
   3.989391428265411
```

## 4. Exotic constraints

The Chebfun `null` function can also handle the more exotic types of boundary conditions that can be imposed in Chebfun (see [1]). For example, suppose we wish to compute the nullspace of the 3rd-order piecewise-smoooth ODE

$$ Lu := 0.1u''' + \sin(x)u'' + u, \quad x\in[-1,1] $$

with the 'boundary' condition

$$ \int(u) = u(0). $$

```matlab
dom = [-1, 1];
L = chebop(@(x, u) .1*diff(u, 3) + sin(x).*diff(u, 2) + u, dom);
L.bc = @(x, u) sum(u) - u(0);
```

Here `null` has no problems!

```matlab
V = null(L)
plot(V, LW, lw), shg
V'*V
```

```text
V =
   chebfun column1 (1 smooth piece)
       interval       length     endpoint values
[      -1,       1]       29      -1.3     0.94
vertical scale = 1.3
   chebfun column2 (1 smooth piece)
       interval       length     endpoint values
[      -1,       1]       32      0.28     -0.8
vertical scale = 1.1
```

![NullSpace figure 07](../../images/ode-eig/NullSpace_07.png)

```matlab
sum(V) - V(0,:)
norm(L(V), 1)
```

```text
ans =
   1.000000000000001  -0.000000000000000
  -0.000000000000000   1.000000000000000
ans =
   1.0e-12 *
  -0.079380946260699   0.124344978758018
ans =
     2.076715816152300e-08
```

## 5. References

1. Chebfun Example [ode-linear/NonstandardBCs](../ode-linear/NonstandardBCs.md)

---

*Translated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); prose and MATLAB code from the original example, copyright The University of Oxford and The Chebfun Developers.  Printed outputs and figures are chebfunjax's.*
