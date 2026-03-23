% interpolation.m — Generate MATLAB references for utils/interpolation module.
% Usage: matlab -batch "addpath('CHEBFUN_PATH'); run('matlab_harness/refs/interpolation.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

rng(42);

ref = struct();

%% ---- cheb_bary_weights (barywts from @chebtech2) ----
for n = [2, 3, 5, 10, 17, 32, 64, 128]
    ref.(sprintf('cheb_barywts_n%d', n)) = chebtech2.barywts(n);
end

%% ---- bary_weights (general) ----
% Use Chebyshev points — general baryWeights should match chebtech2.barywts up to scaling
for n = [5, 10, 17, 32]
    xk = chebpts(n);
    ref.(sprintf('bary_weights_cheb_n%d', n)) = baryWeights(xk);
end

% Random nodes
x_rand10 = sort(2*rand(10,1) - 1);
ref.bary_weights_rand10_nodes = x_rand10;
ref.bary_weights_rand10 = baryWeights(x_rand10);

%% ---- bary (polynomial barycentric interpolation) ----
% Test 1: Interpolate Runge function at Chebyshev nodes, evaluate on grid
for n = [5, 10, 20, 50]
    xk = chebpts(n);
    fk = 1./(1 + 25*xk.^2);
    vk = chebtech2.barywts(n);
    xx = linspace(-1, 1, 200)';
    fx = bary(xx, fk, xk, vk);
    ref.(sprintf('bary_runge_n%d_xx', n)) = xx;
    ref.(sprintf('bary_runge_n%d_fx', n)) = fx;
end

% Test 2: Polynomial exactness — x^4 on 5 Chebyshev points
n = 5;
xk = chebpts(n);
fk = xk.^4;
vk = chebtech2.barywts(n);
xx = linspace(-1, 1, 100)';
fx = bary(xx, fk, xk, vk);
ref.bary_x4_n5_xx = xx;
ref.bary_x4_n5_fx = fx;

% Test 3: Evaluation at the nodes themselves
n = 10;
xk = chebpts(n);
fk = sin(xk);
vk = chebtech2.barywts(n);
fx_at_nodes = bary(xk, fk, xk, vk);
ref.bary_at_nodes_n10 = fx_at_nodes;
ref.bary_at_nodes_n10_fk = fk;

%% ---- trigBary ----
% Test 1: sin(x) on equispaced points in [-pi, pi]
for n = [8, 16, 32]
    xk = trigpts(n, [-pi, pi]);
    fk = sin(xk);
    xx = linspace(-pi + 0.01, pi - 0.01, 100)';
    fx = trigBary(xx, fk, xk, [-pi, pi]);
    ref.(sprintf('trigbary_sin_n%d_xx', n)) = xx;
    ref.(sprintf('trigbary_sin_n%d_fx', n)) = fx;
end

% Test 2: cos(3x) on equispaced points
n = 16;
xk = trigpts(n, [-pi, pi]);
fk = cos(3*xk);
xx = linspace(-pi + 0.01, pi - 0.01, 100)';
fx = trigBary(xx, fk, xk, [-pi, pi]);
ref.trigbary_cos3x_n16_xx = xx;
ref.trigbary_cos3x_n16_fx = fx;

% Test 3: Custom domain [0, 2*pi]
n = 16;
dom = [0, 2*pi];
xk = trigpts(n, dom);
fk = sin(xk);
xx = linspace(0.01, 2*pi - 0.01, 100)';
fx = trigBary(xx, fk, xk, dom);
ref.trigbary_custom_dom_xx = xx;
ref.trigbary_custom_dom_fx = fx;
ref.trigbary_custom_dom_xk = xk;

%% ---- trigBaryWeights ----
for n = [4, 8, 16, 32]
    xk = trigpts(n, [-pi, pi]);
    ref.(sprintf('trig_barywts_eq_n%d', n)) = trigBaryWeights(xk);
end

% Non-equispaced nodes
x_nonunif = [-2.5; -1.2; 0.1; 0.8; 2.3];
ref.trig_barywts_nonunif_nodes = x_nonunif;
ref.trig_barywts_nonunif = trigBaryWeights(x_nonunif);

%% ---- barymat ----
% Test 1: Interpolation matrix for Chebyshev nodes to linspace
for n = [5, 10, 20]
    xk = chebpts(n);
    vk = chebtech2.barywts(n);
    yy = linspace(-1, 1, 30)';
    B = barymat(yy, xk, vk);
    ref.(sprintf('barymat_n%d_m30', n)) = B;
    ref.(sprintf('barymat_n%d_m30_xk', n)) = xk;
    ref.(sprintf('barymat_n%d_m30_yy', n)) = yy;
end

% Test 2: Default weights (Chebyshev 2nd kind)
n = 10;
xk = chebpts(n);
yy = linspace(-1, 1, 20)';
B = barymat(yy, xk);
ref.barymat_default_n10_m20 = B;

save(fullfile(outdir, 'interpolation.mat'), '-struct', 'ref');
fprintf('interpolation.mat: %d fields\n', numel(fieldnames(ref)));
