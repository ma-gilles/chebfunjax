% operators_nonlinear_refs.m — MATLAB references for nonlinear chebop (Newton).
% Pins a Newton-solved nonlinear BVP and a Carrier problem driven from a
% nontrivial initial guess against MATLAB Chebfun at rtol 1e-12.
%
% Usage:
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/operators_nonlinear_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end
ref = struct();

pts = [-0.8; -0.3; 0.1; 0.5; 0.9];
ref.pts = pts;

% NL1: 0.001 u'' - u^3 = 0 on [-1,1], u(-1)=1, u(1)=-1 (Newton-solved).
N = chebop(@(x,u) 0.001*diff(u,2) - u^3, [-1, 1]);
N.lbc = 1; N.rbc = -1;
u1 = N \ 0;
ref.nl1 = feval(u1, pts);

% NL2: Carrier problem, eps=0.01, driven from a nontrivial initial guess
% (selects a specific multi-bump solution of a multi-solution problem).
N = chebop(@(x,u) 0.01*diff(u,2) + 2*(1 - x^2)*u + u^2, [-1, 1]);
N.lbc = 0; N.rbc = 0;
x = chebfun('x', [-1, 1]);
N.init = 2*(x^2 - 1).*(1 - 2./(1 + 20*x^2));
u2 = N \ 1;
ref.nl2 = feval(u2, pts);

save(fullfile(outdir, 'operators_nonlinear.mat'), '-struct', 'ref');
fprintf('operators_nonlinear.mat: %d fields\n', numel(fieldnames(ref)));
