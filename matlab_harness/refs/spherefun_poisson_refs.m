%% MATLAB reference data for the Spherefun Poisson solver.
%
% Cross-validates chebfunjax's Spherefun.poisson (Opus 4.8) against
% MATLAB spherefun.poisson (R2025b), the fast spectral solver for
% laplacian(u) = f on the sphere.  Added by Claude Opus 4.8.
%
% Usage:
%   module load matlab/R2025b
%   cd /home/mg6942/chebfunjax
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/spherefun_poisson_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();
lam = linspace(-3.0, 3.0, 9).';
th  = linspace(0.2, 2.9, 9).';
ref.lam = lam;
ref.th = th;

% RHS from the docstring example (zero mean, exact solution known):
% f = -6*(-1+5*cos(2 th)).*sin(lam).*sin(2 th)
f = spherefun(@(lam,th) -6*(-1+5*cos(2*th)).*sin(lam).*sin(2*th));
u = spherefun.poisson(f, 0, 100);
ref.u_vals = feval(u, lam, th);
ref.u_mean = mean2(u);

save(fullfile(outdir, 'spherefun_poisson.mat'), '-struct', 'ref');
fprintf('Wrote spherefun_poisson.mat with %d fields.\n', numel(fieldnames(ref)));
