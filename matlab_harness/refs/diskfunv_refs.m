% diskfunv_refs.m — MATLAB references for the diskfun vector module (@diskfunv).
% Pins the vector operations on the unit disk at rtol 1e-12 against MATLAB
% Diskfunv: component evaluation, dot, and the pointwise Euclidean magnitude.
%
% Named *_refs.m to avoid colliding with the @diskfunv class.
% Convention: chebfunjax Diskfunv is polar-native, components f(theta,r),
% theta in [-pi,pi], r in [0,1], so MATLAB refs are built with the 'polar'
% flag and evaluated feval(f,theta,r,'polar').  Both chebfunjax and MATLAB
% diskfunv are 2-component, and dot = times(F1,G1)+times(F2,G2) in both.
%
% NOTE ON norm():  chebfunjax Diskfunv.norm() returns the POINTWISE magnitude
%   sqrt(f^2+g^2) as a Diskfun, whereas MATLAB @diskfunv/norm returns the
%   Frobenius SCALAR.  We pin the pointwise field against a MATLAB diskfun of
%   the same magnitude formula, and store the MATLAB Frobenius scalar
%   (norm_frob) for documentation only.  The magnitude field is built from a
%   vector field bounded away from 0 ([2+x, y]) so it is smooth/resolvable.
%
% Usage:
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/diskfunv_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end
ref = struct();

% Matched polar evaluation points (theta in [-pi,pi], r in [0,1)).
tpts = [-2.5; -1.0;  0.0;  1.2;  2.8; -0.5];
rpts = [ 0.2;  0.5;  0.7;  0.3;  0.9;  0.6];
ref.tpts = tpts;
ref.rpts = rpts;

% F = [x; y] = [r cos t; r sin t],  G = [r^2; x] = [r^2; r cos t]
% (components smooth in Cartesian coords, so products resolve cleanly).
f1 = diskfun(@(t,r) r.*cos(t), 'polar');
f2 = diskfun(@(t,r) r.*sin(t), 'polar');
g1 = diskfun(@(t,r) r.^2, 'polar');
g2 = diskfun(@(t,r) r.*cos(t), 'polar');
F = diskfunv(f1, f2);
G = diskfunv(g1, g2);

ref.F1_eval = feval(f1, tpts, rpts, 'polar');
ref.F2_eval = feval(f2, tpts, rpts, 'polar');

% Dot product (scalar field): F.G = x*r^2 + y*x.
ref.dot_eval = feval(dot(F, G), tpts, rpts, 'polar');

% Pointwise magnitude of a bounded-away-from-0 field N = [2+x; y].
n1 = diskfun(@(t,r) 2 + r.*cos(t), 'polar');
n2 = diskfun(@(t,r) r.*sin(t), 'polar');
N = diskfunv(n1, n2);
nmag = diskfun(@(t,r) sqrt((2 + r.*cos(t)).^2 + (r.*sin(t)).^2), 'polar');
ref.norm_eval = feval(nmag, tpts, rpts, 'polar');   % matches chebfunjax N.norm()
ref.norm_frob = norm(N);                            % MATLAB scalar (documentation only)

save(fullfile(outdir, 'diskfunv.mat'), '-struct', 'ref');
fprintf('diskfunv.mat: %d fields\n', numel(fieldnames(ref)));
