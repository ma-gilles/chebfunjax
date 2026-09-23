# Local complexity of a function

*Nick Trefethen, June 2011*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/approx/Local.html)

Python translation: [`examples/approx/local.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/approx/local.py)

Sometimes a function $f$ is more complex in some regions than others. Maryna Kachanovska of the Max Planck Institute in Leipzig suggests the following question about a function $f$ defined on an interval: at each point $x$, how high a degree polynomial do you need to approximate $f$ to a specified accuracy $\varepsilon$ in $[x-d,x+d]$, where $d$ is a small number?

It is easy to compute an answer to such a question with Chebfun, using the syntax `f{x-d,x+d}` to focus on subintervals. For example, here's a function that's quite wiggly in two regions:

```matlab
x = chebfun('x');
f = sin(x./(1.02+cos(5*x)));
```

Let's scan it from left to right, measuring what length of chebfun is needed for a representation to accuracy $10^{-6}$ on intervals of length $0.2$:

```matlab
function Scan(f,ep,d)
  % First, plot the function f:
  FS = 'fontsize'; LW = 'linewidth';
  subplot(2,1,1), plot(f,LW,1.4)
  title('f',FS,14)
  % Next, scan its complexity and make a plot:
  [a,b] = domain(f);
  np = round((b-a)/d);
  xx = linspace(a+d,b-d,np-1);
  chebfunpref.setDefaults('eps',ep);
  ll = 0*xx;
  for j = 1:length(xx)
     ll(j) = length(f{xx(j)-.999999*d,xx(j)+.999999*d});
  end
  subplot(2,1,2), plot(xx,ll,'.-k',LW,1.2)
  xlim([a b])
  title('Local complexity of f',FS,14)
  chebfunpref.setDefaults('factory');
end

Scan(f,1e-6,.04)
```

![Local figure 01](../../images/approx/Local_01.png)

Here is another complicated function and its scan:

```matlab
u = @(ep) chebop(@(x,u) ep*diff(u,2)+x.*cos(x).*u,[-10,10],0)\1;
f = u(.01);
Scan(f,1e-6,.2)
```

![Local figure 02](../../images/approx/Local_02.png)

This last plot seems surprising -- why does the complexity go up at the right endpoint? On closer examination we find that the boundary condition has introduced a blip there:

```matlab
Scan(f{8,10},1e-6,.2)
```

![Local figure 03](../../images/approx/Local_03.png)

```matlab
end
```

---

*Translated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); prose and MATLAB code from the original example, copyright The University of Oxford and The Chebfun Developers.  Printed outputs and figures are chebfunjax's.*
