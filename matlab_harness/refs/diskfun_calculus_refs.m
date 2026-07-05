%% MATLAB reference data for Diskfun calculus (diffx, diffy, laplacian).
%
% Cross-validates chebfunjax's new Diskfun.diffx/diffy/laplacian (Opus
% 4.8) against MATLAB @diskfun (R2025b).  Uses the 2-argument
% diskfun(@(t,r) ...) polar convention matching chebfunjax f(theta, r).
% Added by Claude Opus 4.8.
%
% Usage:
%   module load matlab/R2025b
%   cd /home/mg6942/chebfunjax
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/diskfun_calculus_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();

% interior evaluation points (theta in (-pi,pi), r in (0,1))
theta = linspace(-2.8, 2.8, 9).';
r     = linspace(0.15, 0.9, 9).';
ref.theta = theta;
ref.r = r;

% f = r^2 cos(2 theta)  (= x^2 - y^2)
f = diskfun(@(t,r) r.^2 .* cos(2*t), 'polar');
ref.f_eval = feval(f, theta, r, 'polar');
ref.f_dx = feval(diff(f, 1, 1), theta, r, 'polar');   % d/dx = 2x = 2 r cos t
ref.f_dy = feval(diff(f, 2, 1), theta, r, 'polar');   % d/dy = -2y = -2 r sin t
ref.f_lap = feval(laplacian(f), theta, r, 'polar');    % == 0 (harmonic)

% g = exp(r cos(theta)) sin(r sin(theta))  -- a smooth mixed field
g = diskfun(@(t,r) exp(r.*cos(t)) .* sin(r.*sin(t)), 'polar');
ref.g_eval = feval(g, theta, r, 'polar');
ref.g_dx = feval(diff(g, 1, 1), theta, r, 'polar');
ref.g_dy = feval(diff(g, 2, 1), theta, r, 'polar');
ref.g_lap = feval(laplacian(g), theta, r, 'polar');    % == 0 (real+imag of analytic e^z)

save(fullfile(outdir, 'diskfun_calculus.mat'), '-struct', 'ref');
fprintf('Wrote diskfun_calculus.mat with %d fields.\n', numel(fieldnames(ref)));
