% spherefunv_refs.m — MATLAB references for the spherefun vector module.
% Pins chebfunjax Spherefunv operations at rtol 1e-12 using MATLAB SPHEREFUN
% (scalar) construction — NOT the MATLAB @spherefunv class.
%
% IMPORTANT CONVENTION GAP:  MATLAB @spherefunv is intrinsically a 3-component
%   (Cartesian x,y,z) tangent vector field on the sphere, while chebfunjax
%   Spherefunv is a thin 2-component wrapper [f(lam,theta); g(lam,theta)] whose
%   dot/norm are plain scalar-field algebra (f1*g1+f2*g2, sqrt(f^2+g^2)).  The
%   two classes are therefore NOT directly comparable.  We validate chebfunjax
%   Spherefunv by building the SAME scalar composite fields in MATLAB spherefun
%   and pinning eval/dot/norm against them.  This checks that the chebfunjax
%   wrapper composes the scalar spherefun ops correctly; it does not test
%   MATLAB @spherefunv.
%
% Named *_refs.m to avoid colliding with the @spherefunv class.
% Convention: chebfunjax Spherefun is spherical-native f(lam,theta),
% lam in [-pi,pi] longitude, theta in [0,pi] colatitude, matching MATLAB's
% 2-argument spherefun(@(lam,th) ...), feval(f,lam,th).
%
% NOTE ON norm():  chebfunjax Spherefunv.norm() returns the POINTWISE magnitude
%   sqrt(f^2+g^2) as a Spherefun.  The magnitude field uses a component set
%   bounded away from 0 ([2+x, z]) so it is smooth/resolvable.
%
% Usage:
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/spherefunv_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end
ref = struct();

% Matched spherical evaluation points.
lam = [-2.5; -1.0;  0.0;  1.2;  2.8; -0.5];   % longitude in [-pi,pi]
th  = [ 0.3;  0.8;  1.5;  2.0;  2.7;  1.1];   % colatitude in [0,pi]
ref.lam = lam;
ref.th = th;

% F = [x; z] = [sin(th)cos(lam); cos(th)],  G = [y; x] = [sin(th)sin(lam); sin(th)cos(lam)]
% (components smooth in Cartesian coords -> products resolve cleanly).
f1 = spherefun(@(l,t) sin(t).*cos(l));
f2 = spherefun(@(l,t) cos(t));
g1 = spherefun(@(l,t) sin(t).*sin(l));
g2 = spherefun(@(l,t) sin(t).*cos(l));

ref.F1_eval = feval(f1, lam, th);
ref.F2_eval = feval(f2, lam, th);

% Dot product field:  F.G = x*y + z*x, built as a spherefun of the composite
% (this is exactly what chebfunjax Spherefunv.dot constructs).
ddfun = spherefun(@(l,t) (sin(t).*cos(l)).*(sin(t).*sin(l)) ...
                        + cos(t).*(sin(t).*cos(l)));
ref.dot_eval = feval(ddfun, lam, th);

% Pointwise magnitude of a bounded-away-from-0 field N = [2+x; z].
nmag = spherefun(@(l,t) sqrt((2 + sin(t).*cos(l)).^2 + cos(t).^2));
ref.norm_eval = feval(nmag, lam, th);   % matches chebfunjax N.norm()

save(fullfile(outdir, 'spherefunv.mat'), '-struct', 'ref');
fprintf('spherefunv.mat: %d fields\n', numel(fieldnames(ref)));
