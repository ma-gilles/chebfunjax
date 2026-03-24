% chebcolloc_refs.m — Generate MATLAB references for discretization/chebcolloc module.
%
% Functions tested: ChebColloc2 / ChebColloc1 operations (diffmat, cumsummat,
% points, weights, eval_matrix on [-1,1] and custom domains).
%
% Usage (from jaxchebfun repo root):
%   module load matlab/R2025b
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/chebcolloc_refs.m')"

outdir = fullfile(fileparts(fileparts(mfilename('fullpath'))), '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();

%% --- diffmat (2nd kind) ---
for n = [5, 10, 17, 20, 32]
    ref.(sprintf('diffmat2_n%d', n)) = chebcolloc2.diffmat(n);
    ref.(sprintf('diffmat2_n%d_k2', n)) = chebcolloc2.diffmat(n, 2);
end

%% --- diffmat (1st kind) ---
for n = [5, 10, 17, 20]
    ref.(sprintf('diffmat1_n%d', n)) = chebcolloc1.diffmat(n);
    ref.(sprintf('diffmat1_n%d_k2', n)) = chebcolloc1.diffmat(n, 2);
end

%% --- cumsummat (2nd kind) ---
for n = [5, 10, 17, 20]
    ref.(sprintf('cumsummat2_n%d', n)) = chebcolloc2.cumsummat(n);
end

%% --- cumsummat (1st kind) ---
for n = [5, 10, 17, 20]
    ref.(sprintf('cumsummat1_n%d', n)) = chebcolloc1.cumsummat(n);
end

%% --- collocation points (2nd kind) ---
for n = [5, 10, 17]
    ref.(sprintf('pts2_n%d', n)) = chebtech2.chebpts(n);
end

%% --- collocation points (1st kind) ---
for n = [5, 10, 17]
    ref.(sprintf('pts1_n%d', n)) = chebtech1.chebpts(n);
end

%% --- quadrature weights (2nd kind) ---
for n = [5, 10, 17]
    [~, w] = chebtech2.chebpts(n);
    ref.(sprintf('wts2_n%d', n)) = w;
end

%% --- quadrature weights (1st kind) ---
for n = [5, 10, 17]
    [~, w] = chebtech1.chebpts(n);
    ref.(sprintf('wts1_n%d', n)) = w;
end

%% --- BVP solve: u'' = -1, u(-1)=u(1)=0 -> u = (1-x^2)/2 ---
for n = [5, 10, 20]
    D2 = chebcolloc2.diffmat(n, 2);
    x  = chebtech2.chebpts(n);
    % Enforce BCs by replacing first and last rows
    A = D2;
    A(1,:)   = 0; A(1,1)   = 1;
    A(end,:) = 0; A(end,end) = 1;
    rhs = -ones(n,1);
    rhs(1)   = 0;
    rhs(end) = 0;
    u = A \ rhs;
    ref.(sprintf('bvp_u_n%d', n)) = u;
    ref.(sprintf('bvp_x_n%d', n)) = x;
end

%% --- eval_matrix (interpolation matrix chebcolloc2 -> arbitrary points) ---
n = 10;
y = linspace(-1, 1, 15)';
x = chebtech2.chebpts(n);
w = chebtech2.barywts(n);
B = barymat(y, x, w);
ref.evalmat2_n10_y15 = B;
ref.evalmat2_x10     = x;
ref.evalmat2_y15     = y;

%% --- Same for 1st kind ---
n = 10;
x1 = chebtech1.chebpts(n);
w1 = chebtech1.barywts(n);
B1 = barymat(y, x1, w1);
ref.evalmat1_n10_y15 = B1;
ref.evalmat1_x10     = x1;

%% --- Save ---
save(fullfile(outdir, 'chebcolloc.mat'), '-struct', 'ref');
fprintf('Saved chebcolloc.mat with %d fields.\n', numel(fieldnames(ref)));
