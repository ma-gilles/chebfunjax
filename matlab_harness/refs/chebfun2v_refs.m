% chebfun2v_refs.m — MATLAB references for the chebfun2d vector module (@chebfun2v).
% Pins the 2D vector-calculus operations at rtol 1e-12 against MATLAB Chebfun2v:
%   component evaluation, divergence, curl (2D scalar), dot, cross (2D scalar),
%   Frobenius norm (scalar), and the scalar gradient [f_x; f_y].
%
% Named *_refs.m (NOT chebfun2v.m) to avoid colliding with the @chebfun2v class
% so MATLAB's run() resolves this script.  The saved .mat keeps the plain name.
%
% Coordinate convention: chebfunjax Chebfun2v is Cartesian on [xa,xb]x[ya,yb],
% built from f(x,y), g(x,y), element-wise eval — identical to MATLAB chebfun2v.
% MATLAB diff(.,1,2) = d/dx, diff(.,1,1) = d/dy (as in the scalar chebfun2).
%
% Usage:
%   /usr/licensed/matlab-R2025b/bin/matlab -batch \
%     "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); \
%      run('/home/mg6942/chebfunjax/matlab_harness/refs/chebfun2v_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end
ref = struct();

% Matched evaluation points inside [-1,1]^2 (column vectors, element-wise eval).
px = [-0.9; -0.3;  0.0;  0.4;  0.75; -0.5];
py = [-0.8; -0.1;  0.2;  0.55; 0.9;   0.3];
ref.pts_x = px;
ref.pts_y = py;

% ---- Vector fields on the default domain [-1,1]^2 ----
% F = [cos(x)sin(y); exp(x+y)],  G = [x^2 - y; x*y]
f1 = chebfun2(@(x,y) cos(x).*sin(y));
f2 = chebfun2(@(x,y) exp(x + y));
g1 = chebfun2(@(x,y) x.^2 - y);
g2 = chebfun2(@(x,y) x.*y);
F = chebfun2v(f1, f2);
G = chebfun2v(g1, g2);

% Component evaluation.
ref.F1_eval = feval(f1, px, py);
ref.F2_eval = feval(f2, px, py);
ref.G1_eval = feval(g1, px, py);
ref.G2_eval = feval(g2, px, py);

% Vector calculus (all evaluated at the matched points).
ref.div_eval   = feval(divergence(F), px, py);   % f1_x + f2_y
ref.curl_eval  = feval(curl(F), px, py);         % g_x - f_y (2D scalar curl)
ref.dot_eval   = feval(dot(F, G), px, py);       % F . G
ref.cross_eval = feval(cross(F, G), px, py);     % f1*g2 - f2*g1 (scalar)
ref.normF      = norm(F);                        % Frobenius norm (scalar)

% Scalar gradient of f = cos(x)sin(y):  grad(f) = [f_x; f_y].
[fx, fy] = gradient(f1);
ref.gradx_eval = feval(fx, px, py);              % d/dx cos(x)sin(y) = -sin(x)sin(y)
ref.grady_eval = feval(fy, px, py);              % d/dy cos(x)sin(y) =  cos(x)cos(y)

% ---- Non-unit domain [-2,2] x [0,3] (stresses the diff chain-rule scaling) ----
dom = [-2 2 0 3];
qx = [-1.7; -0.8;  0.0;  1.1;  1.6; -1.2];
qy = [ 0.2;  0.9;  1.5;  2.1;  2.8;  0.6];
ref.dom = dom;
ref.qx = qx;
ref.qy = qy;
h1 = chebfun2(@(x,y) sin(3*x) + cos(2*y), dom);
h2 = chebfun2(@(x,y) x.*y - y.^2, dom);
H = chebfun2v(h1, h2);
ref.div2_eval  = feval(divergence(H), qx, qy);   % 3cos(3x) + (x - 2y)
ref.curl2_eval = feval(curl(H), qx, qy);         % y + 2 sin(2y)

save(fullfile(outdir, 'chebfun2v.mat'), '-struct', 'ref');
fprintf('chebfun2v.mat: %d fields\n', numel(fieldnames(ref)));
