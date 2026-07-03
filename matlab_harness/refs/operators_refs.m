% operators_refs.m — MATLAB references for the operators module (chebop/linop).
% Pins linear BVP solutions and a Dirichlet-Laplacian eigenvalue spectrum
% against MATLAB Chebfun at rtol 1e-12 — the user-facing solver path where a
% plausible-but-wrong answer costs the most.
%
% Usage:
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/operators_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end
ref = struct();

pts = [-0.8; -0.3; 0.1; 0.5; 0.9];
ref.pts = pts;

% L1: u'' = 1 on [-1,1], u(-1)=u(1)=0  (analytic (x^2-1)/2)
L = chebop(@(x,u) diff(u,2), [-1, 1]); L.lbc = 0; L.rbc = 0;
u1 = L \ 1;
ref.u1 = feval(u1, pts);

% L2: u'' - u = x on [-1,1], u(-1)=u(1)=0
L = chebop(@(x,u) diff(u,2) - u, [-1, 1]); L.lbc = 0; L.rbc = 0;
x = chebfun('x', [-1, 1]);
u2 = L \ x;
ref.u2 = feval(u2, pts);

% L3: 0.02 u'' + u' = 1 on [0,1], u(0)=u(1)=0  (advection-diffusion, boundary layer)
pts01 = [0.1; 0.3; 0.5; 0.7; 0.9];
ref.pts01 = pts01;
L = chebop(@(x,u) 0.02*diff(u,2) + diff(u), [0, 1]); L.lbc = 0; L.rbc = 0;
u3 = L \ 1;
ref.u3 = feval(u3, pts01);

% Eigenvalue problem: -u'' on [0,pi] with Dirichlet BCs -> 1,4,9,16,25,36
Le = chebop(@(x,u) -diff(u,2), [0, pi]); Le.bc = 0;
d = eigs(Le, 6);
ref.eig = sort(real(d));

save(fullfile(outdir, 'operators.mat'), '-struct', 'ref');
fprintf('operators.mat: %d fields\n', numel(fieldnames(ref)));
