% autodiff_refs.m -- MATLAB references for the autodiff (ADChebfun / TreeVar) module.
%
% Pins the correctness-critical outputs of Chebfun's automatic Frechet
% differentiation at rtol 1e-12 against MATLAB Chebfun's @adchebfun:
%
%   (i)  VALUE   N(u0)      -- the primal chebfun  w.func, evaluated at points.
%   (ii) JACOBIAN action    -- the Frechet derivative  dN[u0]  applied to a
%        perturbation v, i.e. the chebfun  (w.jacobian * v),  evaluated at
%        the same points.  In MATLAB, `w.jacobian` is an operatorBlock and
%        `w.jacobian * v` (operatorBlock/mtimes -> toFunction) returns the
%        exact analytic action  dN[u0](v)  as an adaptively-resolved chebfun.
%   (iii) LINEARITY          -- w.linearity (1 = linear, 0 = nonlinear), which
%        pins @adchebfun/isLinear and @chebop/isLinear (detect_linearity).
%
% A wrong Jacobian action is the exact failure mode that breaks Newton
% iteration in nonlinear BVP solvers, so any mismatch here is a BUG.
%
% Two batteries:
%   * Differential operators (L1, N1, N2, N3): base u0 = sin(2x).
%   * Unary functions (U_<name>): base sin(2x), or 2+sin(2x) for log/sqrt
%     (which need a positive argument).  These pin the chain-rule multipliers.
% Perturbation for every Jacobian action:  v = cos(3x).
%
% Usage:
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/autodiff_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end
ref = struct();

dom = [-1 1];
ref.dom = dom;

% Evaluation points inside (-1,1) (avoid endpoints).
pts = [-0.9; -0.55; -0.2; 0.15; 0.5; 0.82];
ref.pts = pts;

% Base function and perturbation.
u0 = chebfun(@(x) sin(2*x), dom);         % general base (can be negative)
up = chebfun(@(x) 2 + sin(2*x), dom);     % strictly-positive base (log/sqrt)
v  = chebfun(@(x) cos(3*x), dom);

% ------------------------------------------------------------------
% Battery 1: differential operators.
%   L1: linear, first order       -> linearity 1
%   N1: nonlinear (square - diff)
%   N2: nonlinear (u*u' + sin u)
%   N3: nonlinear (eps*u'' - u^3)  stiff/BVP-like
% ------------------------------------------------------------------
names = {'L1', 'N1', 'N2', 'N3'};
ops   = { @(u) diff(u) - 2*u, ...
          @(u) u.^2 - diff(u), ...
          @(u) u.*diff(u) + sin(u), ...
          @(u) 0.001*diff(u,2) - u.^3 };

for i = 1:numel(ops)
    nm = names{i};
    u  = adchebfun(u0);          % seed identity Jacobian
    w  = ops{i}(u);              % apply operator -> adchebfun
    ref.([nm '_val']) = feval(w.func, pts);          % (i)   VALUE
    Jv = w.jacobian * v;                             % (ii)  JACOBIAN action
    ref.([nm '_jac']) = feval(Jv, pts);
    ref.([nm '_lin']) = double(w.linearity);         % (iii) LINEARITY
end

% ------------------------------------------------------------------
% Battery 2: unary functions  N(u) = f(u).  Order-0 Jacobians
% (diag(f'(u0))) -- pins the chain-rule multiplier for each unary op.
% log/sqrt use the positive base `up`; all others use `u0`.
% ------------------------------------------------------------------
unames = {'sin','cos','tan','exp','sinh','cosh','tanh','log','sqrt'};
ufuns  = { @(u) sin(u), @(u) cos(u), @(u) tan(u), @(u) exp(u), ...
           @(u) sinh(u), @(u) cosh(u), @(u) tanh(u), @(u) log(u), @(u) sqrt(u) };
upos   = [ 0, 0, 0, 0, 0, 0, 0, 1, 1 ];   % 1 => use positive base `up`

for i = 1:numel(ufuns)
    nm = ['U_' unames{i}];
    if upos(i), base = up; else, base = u0; end
    u  = adchebfun(base);
    w  = ufuns{i}(u);
    ref.([nm '_val']) = feval(w.func, pts);
    Jv = w.jacobian * v;
    ref.([nm '_jac']) = feval(Jv, pts);
    ref.([nm '_lin']) = double(w.linearity);   % all nonlinear (0)
end

save(fullfile(outdir, 'autodiff.mat'), '-struct', 'ref');
fprintf('autodiff.mat: %d fields\n', numel(fieldnames(ref)));
