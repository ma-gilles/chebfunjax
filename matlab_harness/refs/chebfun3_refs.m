% chebfun3_refs.m — MATLAB references for the chebfun3d (Chebfun3) module.
% Pins evaluation, sum3 (triple integral), Frobenius norm, and the trilinear
% (Tucker) rank at rtol 1e-12 against MATLAB Chebfun3 (the default chebfun3f
% constructor, i.e. the algorithm chebfunjax translated).
%
% Named *_refs.m (NOT chebfun3.m) to avoid colliding with the @chebfun3 class
% so MATLAB's run() resolves this script.  Saved .mat keeps the plain name.
%
% Usage:
%   /usr/licensed/matlab-R2025b/bin/matlab -batch \
%     "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); \
%      run('/home/mg6942/chebfunjax/matlab_harness/refs/chebfun3_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end
ref = struct();

% Evaluation points inside [-1,1]^3 (column vectors, element-wise eval).
px = [-0.9; -0.3;  0.0;  0.4;  0.75; -0.5;  0.6;  0.1];
py = [-0.8; -0.1;  0.2;  0.55; 0.9;   0.3; -0.4;  0.7];
pz = [ 0.1; -0.2;  0.3; -0.4;  0.5;  -0.6;  0.7; -0.8];
ref.pts_x = px;
ref.pts_y = py;
ref.pts_z = pz;

% Test battery on the default domain [-1,1]^3.
%   f1: separable rank-(1,1,1)
%   f2: moderate-rank rational
%   f3: low-degree polynomial
%   f4: genuinely non-separable higher-rank exponential (stresses ACA/Tucker)
funs = { @(x,y,z) cos(x).*sin(y).*exp(z), ...
         @(x,y,z) 1./(1 + x.^2 + y.^2 + z.^2), ...
         @(x,y,z) x.^2 - y.^2 + z.*x, ...
         @(x,y,z) exp(x.*y.*z) };
for i = 1:numel(funs)
    f = chebfun3(funs{i});
    ref.(sprintf('f%d_eval', i)) = feval(f, px, py, pz);
    ref.(sprintf('f%d_sum3', i)) = sum3(f);
    ref.(sprintf('f%d_norm', i)) = norm(f);          % Frobenius (forward-compat)
    [rX, rY, rZ] = rank(f);
    ref.(sprintf('f%d_rank', i)) = [rX, rY, rZ];     % trilinear (Tucker) rank
end

% Non-unit domain [-2,2] x [0,3] x [-1,2].
dom = [-2 2 0 3 -1 2];
qx = [-1.7; -0.8;  0.0;  1.1;  1.6; -1.2;  0.5;  1.9];
qy = [ 0.2;  0.9;  1.5;  2.1;  2.8;  0.6;  1.3;  2.5];
qz = [-0.7; -0.2;  0.4;  1.0;  1.7; -0.9;  0.8;  1.4];
ref.dom = dom;
ref.qx = qx;
ref.qy = qy;
ref.qz = qz;
g = chebfun3(@(x,y,z) sin(2*x) + cos(y) + x.*z, dom);
ref.g_eval = feval(g, qx, qy, qz);
ref.g_sum3 = sum3(g);
ref.g_norm = norm(g);
[grX, grY, grZ] = rank(g);
ref.g_rank = [grX, grY, grZ];

save(fullfile(outdir, 'chebfun3.mat'), '-struct', 'ref');
fprintf('chebfun3.mat: %d fields\n', numel(fieldnames(ref)));
