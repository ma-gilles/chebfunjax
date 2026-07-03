% spherefun_refs.m — MATLAB references for the spherefun module.
% Named *_refs.m to avoid colliding with Chebfun's @spherefun class.
% chebfunjax Spherefun is spherical-native (f(lam, theta), lam in [-pi,pi]
% longitude, theta in [0,pi] colatitude), matching MATLAB's 2-argument
% spherefun(@(lam,th) ...).  Pins evaluation, sum2 (integral over the sphere)
% and numerical rank at rtol 1e-12.
%
% Usage:
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/spherefun_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end
ref = struct();

lam = [-2.5; -1.0;  0.0;  1.2;  2.8; -0.5];   % longitude in [-pi,pi]
th  = [ 0.3;  0.8;  1.5;  2.0;  2.7;  1.1];   % colatitude in [0,pi]
ref.lam = lam;
ref.th = th;

% Spherical battery (x = sin(th)cos(lam), y = sin(th)sin(lam), z = cos(th)).
sfuns = { @(l,t) cos(t), ...                       % z            (rank 1)
          @(l,t) sin(t).*cos(l), ...               % x            (rank 2)
          @(l,t) exp(sin(t).*cos(l)), ...          % exp(x)       (higher rank)
          @(l,t) 1 + sin(t).*cos(l).*cos(t) };     % 1 + x*z
for i = 1:numel(sfuns)
    f = spherefun(sfuns{i});
    ref.(sprintf('f%d_eval', i)) = feval(f, lam, th);
    ref.(sprintf('f%d_sum', i))  = sum2(f);
    ref.(sprintf('f%d_rank', i)) = rank(f);
end

save(fullfile(outdir, 'spherefun.mat'), '-struct', 'ref');
fprintf('spherefun.mat: %d fields\n', numel(fieldnames(ref)));
