# A parameter dependent ODE with breakpoints

*Asgeir Birkisson, January 2012*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/ode-linear/ParameterODE.html)

Python translation: [`examples/ode-linear/parameter_ode.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/ode-linear/parameter_ode.py)

This example was inspired by a discussion with Paul Constantine [1].

Let the ODE boundary-value problem

$$ (a(x,s)u')' = 1,\qquad u(0) = u(1) = 0, $$

be given, where

$$ a(x,s) = 1+4s(x^2-x) $$

and the prime denotes differentiation with respect to $x$. The exact solution can be shown to be

$$ u(x,s) = {1\over 8s} \log(1+4s(x^2-x)) = {1\over 8s} \log(a(x,s)) . $$

It is clear that for $s=1$, the solution has a singularity at $x=1/2$. Here, we explore what happens when we solve the problem for values of $s$ getting closer and closer to the critical value $s=1$.

## Setting up the problem

We begin by rewriting the differential equation in the form

$$ a(x,s)u'' + a'(x,s)u' = 1, $$

as it will be simpler to work with. We now set up anonymous functions to represent $a$ and $a'$,

```matlab
a =  @(x,s) 1 + 4*s*(x.^2-x);
ap = @(x,s) 4*s*(2*x-1);
```

as well as anonymous functions for the exact solution and the chebfun $x$ on the interval [0,1]:

```matlab
uexact = @(x,s) log(a(x,s)) / (8*s);
chebx = chebfun('x',[0 1]);
```

We can now set up a chebop to represent the boundary-value problem operator. However, since we want to explore what the solution looks like for different values of $s$, we define the chebop as an anonymous function as well (whose output will be a chebop). The two last arguments correspond to imposing homogeneous Dirichlet conditions on the solution.

```matlab
Ns = @(s) chebop(@(x,u) a(x,s).*diff(u,2) + ap(x,s).*diff(u),[0 1], 0, 0);
```

Since we want to take values of $s$ closer and closer to $1$, we rewrite $s$ in the form

$$ s = 1-10^{-\gamma}, $$

where $\gamma$ takes integer values (giving $s=0.9,0.99,0.999,\dots$). We thus define $s$ as an anonymous function

```matlab
s = @(gam) 1-10^(-gam);
```

We can then obtain the solution of the problem for different values of $\gamma$. Again, we use anonymous functions to achieve the desired effect.

```matlab
ugamma = @(g) solvebvp(Ns(s(g)),1);
```

Here, the `solvebvp` method is another way to call the chebop backslash method. The second argument corresponds to the right-hand side of the differential equation.

## Solutions for different values of $\gamma$

We're now all set to solve the problem for different values of $\gamma$.

```matlab
res = []; error = [];
LW = 'linewidth'; FS = 'fontsize';
ax = [0 1 -2.2 0.2];
for g = 1:3
    solgamma = ugamma(g);
    plot(solgamma,LW,1.6)
    ss = sprintf('gamma = %1d    length(solution) = %4d',g,length(solgamma));
    title(ss,FS,12), axis(ax), grid on, snapnow
    res(g) = norm(feval(Ns(s(g)),solgamma)-1);
    error(g) = norm(solgamma - uexact(chebx,s(g)));
end
```

![ParameterODE figure 01](../../images/ode-linear/ParameterODE_01.png)

![ParameterODE figure 02](../../images/ode-linear/ParameterODE_02.png)

![ParameterODE figure 03](../../images/ode-linear/ParameterODE_03.png)

Here we are required to use the `feval` method to evaluate the residual since MATLAB doesn't allowing double indexing, i.e. we can't call `Ns(s(gamma))(solgamma)`.

Values of $\gamma$ up to 3 work fine, but the lenghts of the solutions are increasing.

Looking at the entries in the vector storing the values of the residual reveals that they grow extremely fast with $\gamma$.

```matlab
semilogy(1:3,res,'-*m',LW,1.6), grid on
title('Norm of residual',FS,12), xlabel('\gamma',FS,12)
```

![ParameterODE figure 04](../../images/ode-linear/ParameterODE_04.png)

However, the error remains much better under control:

```matlab
semilogy(1:3,error,'-*r',LW,1.6), grid on
title('Norm of error',FS,12), xlabel('\gamma',FS,12)
```

![ParameterODE figure 05](../../images/ode-linear/ParameterODE_05.png)

## Introducing a breakpoint

The plot above of the solutions for different values of $\gamma$ reveals that the solution gets more and more difficult to represent close to $x= 1/2$ as $\gamma$ increases (i.e., $s$ gets closer to $1$). This makes a good case for introducing a breakpoint in the solution at $x=1/2$, so rather than the solution being represented by a global chebfun, it is represented by two pieces.

We introduce a breakpoint in the operator as follows (notice the second argument to the chebop constructor):

```matlab
Nsbreak = @(s) chebop(@(x,u) a(x,s).*diff(u,2)+ap(x,s).*diff(u),[0 .5 1],0,0);
```

We now redefine the anonymous function which gives the solution.

```matlab
ugammabreak = @(g) solvebvp(Nsbreak(s(g)),1);
```

We're now all set to solve the problem using breakpoints for different values of $\gamma$. Here, values of $\gamma$ up to 6 work with the default chebop settings.

```matlab
chebx = chebfun('x',[0 0.5 1]);
res = []; error = []; legs = [];
for g = 1:7
    solgamma = ugammabreak(g);
    plot(solgamma,LW,1.6)
    ss = sprintf('gamma = %1d    length(solution) = %4d',g,length(solgamma));
    title(ss,FS,12), axis(ax), grid on, snapnow
    res(g) = norm(feval(Nsbreak(s(g)),solgamma)-1);
    error(g) = norm(solgamma - uexact(chebx,s(g)));
end
```

![ParameterODE figure 06](../../images/ode-linear/ParameterODE_06.png)

![ParameterODE figure 07](../../images/ode-linear/ParameterODE_07.png)

![ParameterODE figure 08](../../images/ode-linear/ParameterODE_08.png)

![ParameterODE figure 09](../../images/ode-linear/ParameterODE_09.png)

![ParameterODE figure 10](../../images/ode-linear/ParameterODE_10.png)

![ParameterODE figure 11](../../images/ode-linear/ParameterODE_11.png)

![ParameterODE figure 12](../../images/ode-linear/ParameterODE_12.png)

Again, the errors are quite satisfactory.

```matlab
semilogy(1:7,error,'-*r',LW,1.6), grid on
title('Norm of error',FS,12), xlabel('\gamma',FS,12)
```

![ParameterODE figure 13](../../images/ode-linear/ParameterODE_13.png)

## References

1. Paul Constantine's website: [http://inside.mines.edu/~pconstan/](http://inside.mines.edu/~pconstan/)

---

*Translated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); prose and MATLAB code from the original example, copyright The University of Oxford and The Chebfun Developers.  Printed outputs and figures are chebfunjax's.*
