% quadrature.m — Generate MATLAB references for utils/quadrature module.
% Usage: matlab -batch "addpath('CHEBFUN_PATH'); run('matlab_harness/refs/quadrature.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();

%% Chebyshev points
for n = [5, 10, 17, 32, 64, 128]
    ref.(sprintf('chebpts2_n%d', n)) = chebpts(n);
    ref.(sprintf('chebpts1_n%d', n)) = chebpts(n, 1);
end

%% Legendre (Gauss-Legendre)
for n = [5, 10, 17, 32, 64, 128]
    [x, w] = legpts(n);
    ref.(sprintf('legpts_x_n%d', n)) = x;
    ref.(sprintf('legpts_w_n%d', n)) = w;
end

%% Jacobi (Gauss-Jacobi)
for n = [5, 10, 17, 32, 64]
    for ab = {[0.5, 0.5], [1.0, 1.0], [0.5, 1.5], [2.0, 0.0]}
        a = ab{1}(1); b = ab{1}(2);
        tag = sprintf('jacpts_a%.1f_b%.1f_n%d', a, b, n);
        tag = strrep(tag, '.', 'p');
        [x, w] = jacpts(n, a, b);
        ref.([tag '_x']) = x;
        ref.([tag '_w']) = w;
    end
end

%% Hermite (Gauss-Hermite)
for n = [5, 10, 17, 32, 64]
    [x, w] = hermpts(n);
    ref.(sprintf('hermpts_x_n%d', n)) = x;
    ref.(sprintf('hermpts_w_n%d', n)) = w;
end

%% Laguerre (Gauss-Laguerre)
for n = [5, 10, 17, 32, 64]
    [x, w] = lagpts(n);
    ref.(sprintf('lagpts_x_n%d', n)) = x;
    ref.(sprintf('lagpts_w_n%d', n)) = w;
end

%% Ultraspherical (Gauss-Gegenbauer)
for n = [5, 10, 17, 32, 64]
    for lam = [0.75, 1.5, 2.5]
        tag = sprintf('ultrapts_lam%.2f_n%d', lam, n);
        tag = strrep(tag, '.', 'p');
        [x, w] = ultrapts(n, lam);
        ref.([tag '_x']) = x;
        ref.([tag '_w']) = w;
    end
end

%% Radau (Gauss-Radau)
for n = [3, 5, 10, 17, 32]
    [x, w] = radaupts(n);
    ref.(sprintf('radaupts_x_n%d', n)) = x;
    ref.(sprintf('radaupts_w_n%d', n)) = w;
end

%% Lobatto (Gauss-Lobatto)
for n = [3, 5, 10, 17, 32]
    [x, w] = lobpts(n);
    ref.(sprintf('lobpts_x_n%d', n)) = x;
    ref.(sprintf('lobpts_w_n%d', n)) = w;
end

%% Trigonometric (equispaced)
for n = [4, 8, 16, 32]
    [x, w] = trigpts(n);
    ref.(sprintf('trigpts_x_n%d', n)) = x;
    ref.(sprintf('trigpts_w_n%d', n)) = w;
end

save(fullfile(outdir, 'quadrature.mat'), '-struct', 'ref');
fprintf('quadrature.mat: %d fields\n', numel(fieldnames(ref)));
