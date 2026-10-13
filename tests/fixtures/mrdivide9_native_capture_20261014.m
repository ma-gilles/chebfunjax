function capture(outpath)
root = '/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref';
addpath(root);
addpath(fullfile(root, 'tests', 'chebfun'));
seedRNG(6178);
rng0 = rng;
xr = 2 * rand(100, 1) - 1;
rng1 = rng;

T = restrict(chebpoly(0:3), [-1 -0.5 0 0.5 1]);
L = restrict(legpoly(0:3), [-1 0 1]);
pass = false(1, 9);
pass(1) = normest(T/2 - .5*T) < 10*eps;
x = mrdivide((1:4), T, 'ls');
x0 = feval(x, 0);
pass(2) = abs(x0 - (-2.625)) < 100*eps;
X = mrdivide(eye(4), L, 'ls');
X0 = feval(X, 0);
pass(3) = norm(X0 - [.5 0 -1.25 0].') < 1e2*eps;
x = mrdivide(T, (1:4), 'ls');
x0 = feval(x, 0);
pass(4) = abs(x0 - (-1/15)) < 10*eps;
X = mrdivide(T.', L.', 'ls');
C = diag([1 1 4/3 8/5]);
C(3, 1) = -1/3;
C(4,2) = -3/5;
pass(5) = norm(X - C, inf) < 1e2*eps;

f = chebfun(@(x)sin(20*x)./(x+1), 'exps', [-1 0], 'splitting', 'on');
g = f/3;
op = @(x) sin(20*x)./(3*(x+1));
g_vals = feval(g, xr);
g_exact = op(xr);
err = g_vals - g_exact;
pass(6) = norm(err, inf) < 1e5*vscale(g)*eps;
f = chebfun(@(x)(sin(100*x).^2+1)./(1-x).^0.4, 'exps', [0 -0.4], 'splitting', 'on');
g = mrdivide(5, f, 'ls');
err = abs(g*f - 5);
tol = 100*vscale(g)*eps;
pass(7) = abs(err) < tol;
f = chebfun(@(x)(sin(30*x).^2+1)./(1-x).^0.3, 'exps', [0 -0.3], 'splitting', 'on');
f = f.';
g = chebfun(@(x)9*(sin(30*x).^2+1)./(1-x).^0.3, 'exps', [0 -0.3], 'splitting', 'on');
g = g.';
h = mrdivide(g, f, 'ls');
err = h - 9;
tol = 100*vscale(f)*eps;
pass(8) = abs(err) < tol;

% Literal source clauses 9 operands, with instrumentation only around state and outputs.
dom = [-Inf 3*pi];
domCheck = [-1e6 3*pi];
rng8 = rng;
x = diff(domCheck) * rand(100, 1) + domCheck(1);
rng9 = rng;
op = @(x) [exp(x) x.*exp(x) (1-exp(x))./x];
A = chebfun(op, dom);
B = 3;
X = mrdivide(A, B, 'ls');
opExact = @(x) [exp(x)/3 x.*exp(x)/3 (1-exp(x))./(3*x)];
XVals = feval(X, x);
XExact = opExact(x);
err = XVals - XExact;
Xvscale = get(X, 'vscale');
bound = 1e2*max(eps*Xvscale);
pass(9) = norm(err, inf) < bound;
metadata = struct('matlab_release', version('-release'), 'seed', 6178, ...
    'domain', dom, 'domain_check', domCheck, 'source_bound', bound, ...
    'x_size', size(x), 'xvals_size', size(XVals), 'xexact_size', size(XExact), ...
    'err_size', size(err), 'vscale_size', size(Xvscale));
save(outpath, 'rng0', 'rng1', 'rng8', 'rng9', 'xr', 'x', 'XVals', ...
    'XExact', 'err', 'Xvscale', 'bound', 'pass', 'metadata', '-v7');
end
