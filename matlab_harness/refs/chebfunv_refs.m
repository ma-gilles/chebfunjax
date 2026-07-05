%% MATLAB reference data for Chebfun2v / Chebfun3v vector fields.
%
% Cross-validates chebfunjax's Chebfun2v (div, curl, dot, cross) and
% Chebfun3v (dot, cross, norm) against MATLAB @chebfun2v / @chebfun3v
% (R2025b).  Added by Claude Opus 4.8.
%
% Usage:
%   module load matlab/R2025b
%   cd /home/mg6942/chebfunjax
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/chebfunv_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();

xv = linspace(-0.8, 0.8, 5);
yv = linspace(-0.6, 0.6, 5);
[X, Y] = meshgrid(xv, yv);
ref.X = X;  ref.Y = Y;

% ---- Chebfun2v: F = [x^2 y, sin(x) y] ----
F = chebfun2v(@(x,y) x.^2.*y, @(x,y) sin(x).*y);
ref.f2v_div  = feval(divergence(F), X, Y);   % 2xy + sin(x)
ref.f2v_curl = feval(curl(F), X, Y);         % cos(x) y - x^2

% ---- Chebfun2v dot / cross with G = [y, x] ----
G = chebfun2v(@(x,y) y, @(x,y) x);
ref.f2v_dot = feval(dot(F, G), X, Y);        % x^2 y * y + sin(x) y * x
% 2D cross -> scalar
ref.f2v_cross = feval(cross(F, G), X, Y);

% ---- Chebfun3v: H = [x y, y z, x z] ; dot/cross/norm ----
P = [ -0.6 -0.3  0.2;
       0.1  0.5 -0.4;
       0.7 -0.7  0.6;
      -0.2  0.4  0.8 ];
ref.P = P;
xx = P(:,1); yy = P(:,2); zz = P(:,3);
H = chebfun3v(@(x,y,z) x.*y, @(x,y,z) y.*z, @(x,y,z) x.*z);
K = chebfun3v(@(x,y,z) z, @(x,y,z) x, @(x,y,z) y);
ref.f3v_dot  = feval(dot(H, K), xx, yy, zz);
C3 = cross(H, K);
ref.f3v_cross1 = feval(C3(1), xx, yy, zz);
ref.f3v_cross2 = feval(C3(2), xx, yy, zz);
ref.f3v_cross3 = feval(C3(3), xx, yy, zz);

save(fullfile(outdir, 'chebfunv.mat'), '-struct', 'ref');
fprintf('Wrote chebfunv.mat with %d fields.\n', numel(fieldnames(ref)));
