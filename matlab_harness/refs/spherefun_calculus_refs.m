%% MATLAB reference data for Spherefun calculus (laplacian, diff, grad).
%
% Directly cross-validates chebfunjax's new Spherefun.laplacian / diff /
% grad (Opus 4.8) against MATLAB @spherefun (R2025b), rather than only
% against exact harmonic identities.  Uses the 2-argument
% spherefun(@(lam,th) ...) convention that matches chebfunjax
% f(lam, theta).  Added by Claude Opus 4.8.
%
% Usage:
%   module load matlab/R2025b
%   cd /home/mg6942/chebfunjax
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/spherefun_calculus_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();

% Evaluation points (lam in (-pi,pi), th in (0,pi)).
lam = linspace(-3.0, 3.0, 9).';
th  = linspace(0.2, 2.9, 9).';
ref.lam = lam;
ref.th  = th;

% ---- f = sin(th)^2 cos(2 lam)  (a real spherical harmonic, degree 2) ----
f = spherefun(@(lam,th) sin(th).^2 .* cos(2*lam));
ref.f_eval = feval(f, lam, th);
ref.f_lap  = feval(laplacian(f), lam, th);   % == -6 f  (l=2)
ref.f_dx   = feval(diff(f, 1), lam, th);      % d/dx (tangential)
ref.f_dy   = feval(diff(f, 2), lam, th);      % d/dy
ref.f_dz   = feval(diff(f, 3), lam, th);      % d/dz

% ---- g = cos(th)  (= z, degree 1) ----
g = spherefun(@(lam,th) cos(th));
ref.g_lap = feval(laplacian(g), lam, th);     % == -2 g
ref.g_dz  = feval(diff(g, 3), lam, th);       % -sin(th)^2

save(fullfile(outdir, 'spherefun_calculus.mat'), '-struct', 'ref');
fprintf('Wrote spherefun_calculus.mat with %d fields.\n', numel(fieldnames(ref)));
