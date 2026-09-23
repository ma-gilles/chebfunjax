# A periodic ODE system

*Nick Hale, December 2014*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/ode-linear/PeriodicSystem.html)

Python translation: [`examples/ode-linear/periodic_system.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/ode-linear/periodic_system.py)

Chebfun can solve systems of ODEs with periodic boundary conditions. For example, consider the equations

$$ u - v' = 0, \qquad u'' + v = \cos(x) $$

on the interval $[-\pi, \pi]$ with periodic boundary conditions on $u$ and $v$. A Chebfun solution could be put together like this:

```matlab
d = [-pi,pi];
A = chebop(d);
A.op = @(x,u,v) [u-diff(v); diff(u,2)+v];
x = chebfun('x',d);
f = [0;cos(x)];
A.bc = 'periodic';
u = A\f;
u{1}, u{2}
```

```text
ans =
   chebfun column (1 smooth piece)
       interval       length     endpoint values trig
[    -3.1,     3.1]        3       0.5      0.5
vertical scale = 0.68
ans =
   chebfun column (1 smooth piece)
       interval       length     endpoint values trig
[    -3.1,     3.1]        3      -0.5     -0.5
vertical scale = 0.68
```

Because the boundardy conditions are periodic, the system of ODEs is solved with a Fourier collocation method, and the solution $u$ is represented by a Fourier series. (This is what `trig` means in the display of $u$ above.) We plot the result:

```matlab
LW = 'linewidth'; lw = 2; FS = 'fontsize'; fs = 14;
plot(u,LW,lw), title('Solutions u and v',FS,fs), legend('u','v');
```

![PeriodicSystem figure 01](../../images/ode-linear/PeriodicSystem_01.png)

For this problem, the solution can actually be computed analytically. How close were we?

```matlab
exact = [cos(x+3*pi/4) cos(x+pi/4)]/sqrt(2);
err = max([norm(u{1}-exact(:,1),inf) norm(u{2}-exact(:,2),inf)])
```

```text
err =
     3.390077316427179e-16
```

We show this also works for piecewise problems by artificially introducing a breakpoint at the origin.

```matlab
A.domain = [-pi,0,pi];
u = A\f;
u{1}, u{2}
```

```text
ans =
   chebfun column (2 smooth pieces)
       interval       length     endpoint values
[    -3.1,       0]       17       0.5     -0.5
[       0,     3.1]       18      -0.5      0.5
vertical scale = 0.7    Total length = 35
ans =
   chebfun column (2 smooth pieces)
       interval       length     endpoint values
[    -3.1,       0]       17      -0.5      0.5
[       0,     3.1]       18       0.5     -0.5
vertical scale = 0.7    Total length = 35
```

The solution is now represented by a Chebyshev series, and the equation has been solved with a Chebyshev collocation method, because Fourier collocation methods can't handle breakpoints.

```matlab
plot(u,LW,lw), title('Solutions u and v',FS,fs), legend('u','v');
err = max([norm(u{1}-exact(:,1),inf) norm(u{2}-exact(:,2),inf)])
```

```text
err =
     7.550883067076701e-14
```

![PeriodicSystem figure 02](../../images/ode-linear/PeriodicSystem_02.png)

---

*Translated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); prose and MATLAB code from the original example, copyright The University of Oxford and The Chebfun Developers.  Printed outputs and figures are chebfunjax's.*
