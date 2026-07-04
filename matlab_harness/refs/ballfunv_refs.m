% ballfunv_refs.m — MATLAB references for the ballfun vector module (@ballfunv).
% Pins the 3D vector operations on the unit ball at rtol 1e-12 against MATLAB
% Ballfunv: component evaluation, dot, cross, and the (scalar) L2/Frobenius norm.
%
% Named *_refs.m to avoid colliding with the @ballfunv class.
% Convention: chebfunjax Ballfunv is built Cartesian (op(x,y,z), default) and
% evaluated in spherical coords f(r, lam, th) with x=r sin(th)cos(lam),
% y=r sin(th)sin(lam), z=r cos(th) (r in [0,1], lam in [-pi,pi], th in [0,pi]).
% __call__ does a meshgrid eval; a single matched point is read as [0,0,0].
% ballfunv/norm returns a scalar (matches chebfunjax Ballfunv.norm() -> float).
%
% Usage:
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/ballfunv_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end
ref = struct();

rp   = [0.2;  0.5;  0.7;  0.9;  0.3;  0.6];   % radius in [0,1]
lamp = [-2.5; -1.0;  0.0;  1.2;  2.8; -0.5];  % longitude in [-pi,pi]
thp  = [0.3;  0.8;  1.5;  2.0;  2.7;  1.1];   % colatitude in [0,pi]
ref.rp = rp;
ref.lamp = lamp;
ref.thp = thp;

% F = [z; x; 1+x*z],  G = [x; z; 1]   (Cartesian components)
f1 = ballfun(@(x,y,z) z);
f2 = ballfun(@(x,y,z) x);
f3 = ballfun(@(x,y,z) 1 + x.*z);
g1 = ballfun(@(x,y,z) x);
g2 = ballfun(@(x,y,z) z);
g3 = ballfun(@(x,y,z) 1 + 0*x);
F = ballfunv(f1, f2, f3);
G = ballfunv(g1, g2, g3);

% Helper: evaluate a ballfun at the matched (r,lam,th) points in spherical coords.
evalpts = @(f) arrayfun(@(k) feval(f, rp(k), lamp(k), thp(k), 'spherical'), ...
                        (1:numel(rp))');

ref.F1_eval = evalpts(f1);
ref.F2_eval = evalpts(f2);
ref.F3_eval = evalpts(f3);

% Dot product (scalar field on the ball).
dd = dot(F, G);
ref.dot_eval = evalpts(dd);

% Cross product (vector field, 3 components).
H = cross(F, G);
Hc = H.comp;
ref.crossx_eval = evalpts(Hc{1});
ref.crossy_eval = evalpts(Hc{2});
ref.crossz_eval = evalpts(Hc{3});

% L2/Frobenius norm (scalar) — matches chebfunjax Ballfunv.norm().
ref.normF = norm(F);

save(fullfile(outdir, 'ballfunv.mat'), '-struct', 'ref');
fprintf('ballfunv.mat: %d fields\n', numel(fieldnames(ref)));
