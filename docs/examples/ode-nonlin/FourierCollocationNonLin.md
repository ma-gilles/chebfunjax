# Nonlinear Periodic ODE

*Hadrien Montanelli, December 2014*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/ode-nonlin/FourierCollocationNonLin.html)

Python translation: [`examples/ode-nonlin/fourier_collocation_nonlin.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/ode-nonlin/fourier_collocation_nonlin.py)

```matlab
LW = 'linewidth'; dom = [0 2*pi];
```

Chebfun uses Fourier collocation to solve linear, nonlinear and systems of ODEs with periodic boundary conditions. Consider the nonlinear ODE

$$ u' - u\cos(u) = \cos(4x), $$

on $[0, 2\pi]$, with periodic boundary conditions. For nonlinear ODEs, we need to specify an intial guess; let us try $\cos(x)$. We can solve the ODE in Chebfun as follows.

```matlab
f = chebfun(@(x) cos(4*x), dom);
N = chebop(@(u) diff(u) - u.*cos(u), dom);
N.bc = 'periodic';
N.init = chebfun(@(x) cos(x), dom);
u = N \ f
```

```text
u =
   chebfun column (1 smooth piece)
       interval       length     endpoint values trig
[       0,     6.3]       73    -0.058   -0.058
vertical scale = 0.24
```

Let us plot the initial guess in dashed blue, and the solution in blue.

```matlab
figure, plot(N.init, '--b', LW, 2)
hold on, plot(u, 'b', LW, 2)
```

![FourierCollocationNonLin figure 01](../../images/ode-nonlin/FourierCollocationNonLin_01.png)

The solution $u(x)$ satisfies the ODE to high accuracy:

```matlab
norm(N*u - f, inf)
```

```text
ans =
     7.563761544447662e-14
```

If we start with another initial guess, we might obtain another solution. Let us try $\sin(x)^2$, plot it in dashed green, and plot the solution in green.

```matlab
N.init = chebfun(@(x) sin(x).^2, dom);
v = N \ f
hold on, plot(N.init, '--g', LW, 2)
hold on, plot(v, 'g', LW, 2)
```

```text
v =
   chebfun column (1 smooth piece)
       interval       length     endpoint values trig
[       0,     6.3]       81       1.6      1.6
vertical scale = 1.8
```

![FourierCollocationNonLin figure 02](../../images/ode-nonlin/FourierCollocationNonLin_02.png)

The solution $v(x)$ satisfies the ODE to high accuracy too:

```matlab
norm(N*v - f, inf)
```

```text
ans =
     7.756028661427450e-13
```

For nonlinear ODEs, the relationships between intial guesses and solutions are difficult to analyse. In this example, we chose two different guesses, and this led to two different solutions.

---

*Translated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); prose and MATLAB code from the original example, copyright The University of Oxford and The Chebfun Developers.  Printed outputs and figures are chebfunjax's.*
