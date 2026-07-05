%% Nonlinear Chebop BVP reference data (2nd batch).
%
% Cross-validates chebfunjax's nonlinear Chebop.solve (damped Newton)
% against MATLAB @chebop (R2025b) on the Bratu equation (lower branch)
% and a monotone sinh-reaction BVP.  Added by Claude Opus 4.8.
%
% Usage:
%   module load matlab/R2025b
%   cd /home/mg6942/chebfunjax
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/chebop_nonlinear2_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();
pts = linspace(0.05, 0.95, 13).';
ref.pts = pts;
pts2 = linspace(-0.9, 0.9, 13).';
ref.pts2 = pts2;

% ---- Bratu: u'' + exp(u) = 0, u(0)=u(1)=0 (lower branch) ----
Lb = chebop(@(x,u) diff(u,2) + exp(u), [0, 1]);
Lb.lbc = 0; Lb.rbc = 0;
ub = Lb \ 0;
ref.bratu_vals = feval(ub, pts);

% ---- monotone sinh-reaction: u'' - sinh(u) = x, u(-1)=u(1)=0 ----
Ls = chebop(@(x,u) diff(u,2) - sinh(u), [-1, 1]);
Ls.lbc = 0; Ls.rbc = 0;
us = Ls \ chebfun(@(x) x, [-1, 1]);
ref.sinh_vals = feval(us, pts2);

save(fullfile(outdir, 'chebop_nonlinear2.mat'), '-struct', 'ref');
fprintf('Wrote chebop_nonlinear2.mat with %d fields.\n', numel(fieldnames(ref)));
