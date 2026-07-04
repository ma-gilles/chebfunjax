% ballfun_calculus_refs.m — MATLAB references for the new Ballfun calculus
% operations (diff in Cartesian directions, laplacian, grad components),
% pinned at probe points inside the unit ball.
%
% Usage:
%   module load matlab/R2025b
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/ballfun_calculus_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end
ref = struct();

% Probe points in spherical coordinates (r, lambda, theta)
r  = [0.20; 0.45; 0.60; 0.85];
lm = [0.30; -1.20; 2.50; -2.90];
th = [0.70; 1.30; 2.10; 0.40];
ref.r = r; ref.lm = lm; ref.th = th;

% Case 1: polynomial f = x.*y + z.^2
f1 = ballfun(@(x,y,z) x.*y + z.^2);
d1x = diff(f1, 1); d1y = diff(f1, 2); d1z = diff(f1, 3);
l1 = laplacian(f1);
for i = 1:numel(r)
    ref.f1_dx(i,1) = feval(d1x, r(i), lm(i), th(i), 'spherical');
    ref.f1_dy(i,1) = feval(d1y, r(i), lm(i), th(i), 'spherical');
    ref.f1_dz(i,1) = feval(d1z, r(i), lm(i), th(i), 'spherical');
    ref.f1_lap(i,1) = feval(l1, r(i), lm(i), th(i), 'spherical');
end

% Case 2: smooth transcendental f = sin(2x).*exp(y).*cos(z)
f2 = ballfun(@(x,y,z) sin(2*x).*exp(y).*cos(z));
d2x = diff(f2, 1); d2y = diff(f2, 2);
l2 = laplacian(f2);
for i = 1:numel(r)
    ref.f2_dx(i,1) = feval(d2x, r(i), lm(i), th(i), 'spherical');
    ref.f2_dy(i,1) = feval(d2y, r(i), lm(i), th(i), 'spherical');
    ref.f2_lap(i,1) = feval(l2, r(i), lm(i), th(i), 'spherical');
end

% Case 3: second derivative d2/dz2 of x.^2.*z.^3
f3 = ballfun(@(x,y,z) x.^2.*z.^3);
d3zz = diff(f3, 3, 2);
for i = 1:numel(r)
    ref.f3_dzz(i,1) = feval(d3zz, r(i), lm(i), th(i), 'spherical');
end

save(fullfile(outdir, 'ballfun_calculus.mat'), '-struct', 'ref');
fprintf('ballfun_calculus.mat written with %d fields\n', numel(fieldnames(ref)));
