% chebfun3v_refs.m — MATLAB references for the chebfun3d vector module (@chebfun3v).
% Pins the available 3D vector operations at rtol 1e-12 against MATLAB Chebfun3v:
%   component evaluation, dot, cross, and the pointwise Euclidean magnitude.
%
% NOTE ON norm():  chebfunjax Chebfun3v.norm() returns the POINTWISE magnitude
%   sqrt(f^2+g^2+h^2) as a Chebfun3 (a scalar field), whereas MATLAB
%   @chebfun3v/norm returns the Frobenius SCALAR sqrt(sum norm(comp)^2).
%   These are different quantities.  We therefore pin the chebfunjax pointwise
%   field against a MATLAB chebfun3 built from the same magnitude formula, and
%   ALSO store the MATLAB Frobenius scalar (normF_frob) for documentation only.
%
% Named *_refs.m to avoid colliding with the @chebfun3v class.
% Coordinate convention: Cartesian on [-1,1]^3, element-wise eval f(x,y,z).
%
% Usage:
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/chebfun3v_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end
ref = struct();

% Matched evaluation points inside [-1,1]^3 (column vectors, element-wise eval).
px = [-0.9; -0.3;  0.0;  0.4;  0.75; -0.5];
py = [-0.8; -0.1;  0.2;  0.55; 0.9;   0.3];
pz = [ 0.1; -0.2;  0.3; -0.4;  0.5;  -0.6];
ref.pts_x = px;
ref.pts_y = py;
ref.pts_z = pz;

% F = [cos(x)sin(y); exp(z); x*y]   (exp(z) keeps |F| bounded away from 0, so
%                                    the magnitude field is smooth/resolvable)
% G = [x*z; y; z^2]
f1 = chebfun3(@(x,y,z) cos(x).*sin(y));
f2 = chebfun3(@(x,y,z) exp(z));
f3 = chebfun3(@(x,y,z) x.*y);
g1 = chebfun3(@(x,y,z) x.*z);
g2 = chebfun3(@(x,y,z) y);
g3 = chebfun3(@(x,y,z) z.^2);
F = chebfun3v(f1, f2, f3);
G = chebfun3v(g1, g2, g3);

% Component evaluation.
ref.F1_eval = feval(f1, px, py, pz);
ref.F2_eval = feval(f2, px, py, pz);
ref.F3_eval = feval(f3, px, py, pz);

% Dot product (scalar field).
ref.dot_eval = feval(dot(F, G), px, py, pz);

% Cross product (vector field, 3 components).
C = cross(F, G);
Cc = C.components;
ref.crossx_eval = feval(Cc{1}, px, py, pz);
ref.crossy_eval = feval(Cc{2}, px, py, pz);
ref.crossz_eval = feval(Cc{3}, px, py, pz);

% Pointwise magnitude sqrt(f^2+g^2+h^2) as a chebfun3 (matches chebfunjax norm()).
nf = chebfun3(@(x,y,z) sqrt((cos(x).*sin(y)).^2 + exp(z).^2 + (x.*y).^2));
ref.normP_eval = feval(nf, px, py, pz);

% MATLAB @chebfun3v/norm Frobenius scalar (documentation only; different semantics).
ref.normF_frob = norm(F);

save(fullfile(outdir, 'chebfun3v.mat'), '-struct', 'ref');
fprintf('chebfun3v.mat: %d fields\n', numel(fieldnames(ref)));
