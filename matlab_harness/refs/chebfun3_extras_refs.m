%% Expanded MATLAB reference data for Chebfun3 operations.
%
% Cross-validates chebfunjax's Chebfun3 against MATLAB @chebfun3
% (R2025b): evaluation, triple integral, partial derivatives, and a
% Laplacian identity.  Added by Claude Opus 4.8.
%
% Usage:
%   module load matlab/R2025b
%   cd /home/mg6942/chebfunjax
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/chebfun3_extras_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();

% Deterministic evaluation points inside [-1,1]^3 (columns x,y,z).
P = [ -0.6 -0.3  0.2;
       0.1  0.5 -0.4;
       0.7 -0.7  0.6;
      -0.2  0.4  0.8;
       0.5  0.0 -0.9 ];
ref.P = P;
xx = P(:,1); yy = P(:,2); zz = P(:,3);

% ---- f = exp(x) sin(y) cos(z) ----
f = chebfun3(@(x,y,z) exp(x).*sin(y).*cos(z));
ref.f_eval = feval(f, xx, yy, zz);
ref.f_sum3 = sum3(f);
ref.f_dx = feval(diff(f, 1, 1), xx, yy, zz);
ref.f_dy = feval(diff(f, 1, 2), xx, yy, zz);
ref.f_dz = feval(diff(f, 1, 3), xx, yy, zz);

% ---- g = x^2 + y^2 + z^2 ; Laplacian == 6 everywhere ----
g = chebfun3(@(x,y,z) x.^2 + y.^2 + z.^2);
ref.g_eval = feval(g, xx, yy, zz);
ref.g_sum3 = sum3(g);   % 3 * (2/3) * 2 * 2 = 8
ref.g_lap = feval(diff(g,2,1) + diff(g,2,2) + diff(g,2,3), xx, yy, zz);

save(fullfile(outdir, 'chebfun3_extras.mat'), '-struct', 'ref');
fprintf('Wrote chebfun3_extras.mat with %d fields.\n', numel(fieldnames(ref)));
