% chebfun2d.m — MATLAB references for the chebfun2d (Chebfun2) module.
% Pins evaluation, sum2 (double integral), Frobenius norm, and numerical rank
% at rtol 1e-12 against MATLAB Chebfun2 (the least unit-tested layer).
%
% Usage:
%   module load matlab/R2025b
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/chebfun2d.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end
ref = struct();

% Evaluation points inside [-1,1]^2 (column vectors, element-wise eval).
px = [-0.9; -0.3;  0.0;  0.4;  0.75; -0.5;  0.6;  0.1];
py = [-0.8; -0.1;  0.2;  0.55; 0.9;   0.3; -0.4;  0.7];
ref.pts_x = px;
ref.pts_y = py;

% Test battery on the default domain [-1,1]^2.
funs = { @(x,y) cos(x).*sin(y), ...
         @(x,y) exp(x + y), ...
         @(x,y) 1./(1 + x.^2 + 2*y.^2), ...
         @(x,y) x.^2 - y.^2 + x.*y };
for i = 1:numel(funs)
    f = chebfun2(funs{i});
    ref.(sprintf('f%d_eval', i)) = feval(f, px, py);
    ref.(sprintf('f%d_sum2', i)) = sum2(f);
    ref.(sprintf('f%d_norm', i)) = norm(f);
    ref.(sprintf('f%d_rank', i)) = rank(f);
end

% Non-unit domain [-2,2] x [0,3].
dom = [-2 2 0 3];
qx = [-1.7; -0.8;  0.0;  1.1;  1.6; -1.2;  0.5;  1.9];
qy = [ 0.2;  0.9;  1.5;  2.1;  2.8;  0.6;  1.3;  2.5];
ref.dom = dom;
ref.qx = qx;
ref.qy = qy;
g = chebfun2(@(x,y) sin(3*x) + cos(2*y), dom);
ref.g_eval = feval(g, qx, qy);
ref.g_sum2 = sum2(g);
ref.g_norm = norm(g);
ref.g_rank = rank(g);

save(fullfile(outdir, 'chebfun2d.mat'), '-struct', 'ref');
fprintf('chebfun2d.mat: %d fields\n', numel(fieldnames(ref)));
