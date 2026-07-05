%% MATLAB reference data for the Diskfun Poisson solver.
%
% Cross-validates chebfunjax's Diskfun.poisson (Opus 4.8) against MATLAB
% diskfun.poisson (R2025b) with homogeneous Dirichlet BC.  Added by
% Claude Opus 4.8.
%
% Usage:
%   module load matlab/R2025b
%   cd /home/mg6942/chebfunjax
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/diskfun_poisson_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();
theta = linspace(-2.8, 2.8, 9).';
r     = linspace(0.15, 0.9, 9).';
ref.theta = theta;
ref.r = r;

bc = @(th) 0*th;

% RHS 1: f = r^2 cos(2 theta) (smooth), homogeneous Dirichlet
f1 = diskfun(@(t,r) r.^2 .* cos(2*t), 'polar');
u1 = diskfun.poisson(f1, bc, 64);
ref.u1_vals = feval(u1, theta, r, 'polar');

% RHS 2: manufactured f = -8 r^2 cos(2 theta) -> exact u=(1-r^2) r^2 cos2t
f2 = diskfun(@(t,r) -12 * r.^2 .* cos(2*t), 'polar');
u2 = diskfun.poisson(f2, bc, 64);
ref.u2_vals = feval(u2, theta, r, 'polar');

save(fullfile(outdir, 'diskfun_poisson.mat'), '-struct', 'ref');
fprintf('Wrote diskfun_poisson.mat with %d fields.\n', numel(fieldnames(ref)));
