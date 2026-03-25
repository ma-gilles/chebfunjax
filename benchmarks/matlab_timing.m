% Run in MATLAB with Chebfun on path
% Usage: module load matlab/R2025b && matlab -batch "run('benchmarks/matlab_timing.m')"
addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref');

fprintf('=== MATLAB Chebfun Timing Benchmarks ===\n');
fprintf('MATLAB version: %s\n', version);

% ---- 1D: Construction of sin(x) ----
N = 100;
tic;
for i = 1:N
    f = chebfun(@sin);
end
t_construct = toc / N * 1000;

% ---- 1D: Evaluation at a single point ----
f = chebfun(@sin);
x = 0.5;
N_eval = 10000;
tic;
for i = 1:N_eval
    y = f(x);
end
t_eval = toc / N_eval * 1000;

% ---- 1D: Differentiation ----
f_diff_input = chebfun(@(x) sin(10*x));
N_diff = 1000;
tic;
for i = 1:N_diff
    fp = diff(f_diff_input);
end
t_diff = toc / N_diff * 1000;

% ---- 1D: Integration (sum) ----
f_sum_input = chebfun(@(x) exp(-x.^2));
N_sum = 1000;
tic;
for i = 1:N_sum
    s = sum(f_sum_input);
end
t_sum = toc / N_sum * 1000;

% ---- 1D: Roots ----
f_roots_input = chebfun(@(x) sin(5*pi*x));
N_roots = 100;
tic;
for i = 1:N_roots
    r = roots(f_roots_input);
end
t_roots = toc / N_roots * 1000;

% ---- 1D: Evaluation at 1000 points ----
f_eval1000 = chebfun(@sin);
xs = linspace(-1, 1, 1000)';
% Warm up
warmup_val = f_eval1000(xs);
N_eval1000 = 500;
tic;
for i = 1:N_eval1000
    ys = f_eval1000(xs);
end
t_eval1000 = toc / N_eval1000 * 1000;

% ---- 1D: Scaling — construction for high-degree functions ----
% Approximate functions needing ~n Chebyshev coefficients by using fixed-n
degrees = [10, 100, 1000];
t_construct_n = zeros(size(degrees));
for di = 1:length(degrees)
    n = degrees(di);
    tic;
    for rep = 1:20
        fn = chebfun(@sin, 'trunc', n);
    end
    t_construct_n(di) = toc / 20 * 1000;
end

% ---- 2D: Construction of cos(x+y) Chebfun2 ----
N_c2 = 20;
tic;
for i = 1:N_c2
    f2 = chebfun2(@(x,y) cos(x+y));
end
t_chebfun2_construct = toc / N_c2 * 1000;

% ---- 2D: Evaluation of Chebfun2 at 1000 points ----
f2 = chebfun2(@(x,y) cos(x+y));
xp = linspace(-1, 1, 1000)';
yp = linspace(-1, 1, 1000)';
warmup_val2 = f2(xp, yp);  % warm up
N_e2 = 200;
tic;
for i = 1:N_e2
    v2 = f2(xp, yp);
end
t_chebfun2_eval = toc / N_e2 * 1000;

% ---- 2D: diff (partial derivative) ----
f2d = chebfun2(@(x,y) exp(x.*y));
N_d2 = 200;
tic;
for i = 1:N_d2
    df2 = diff(f2d, 1, 2);
end
t_chebfun2_diff = toc / N_d2 * 1000;

% ---- 2D: sum2 (double integral) ----
f2s = chebfun2(@(x,y) cos(x+y));
N_s2 = 1000;
tic;
for i = 1:N_s2
    s2 = sum2(f2s);
end
t_chebfun2_sum2 = toc / N_s2 * 1000;

% ---- 2D: norm ----
f2n = chebfun2(@(x,y) cos(x+y));
N_n2 = 500;
tic;
for i = 1:N_n2
    nrm2 = norm(f2n);
end
t_chebfun2_norm = toc / N_n2 * 1000;

% ---- 3D: Construction of cos(x+y+z) Chebfun3 ----
N_c3 = 5;
tic;
for i = 1:N_c3
    f3 = chebfun3(@(x,y,z) cos(x+y+z));
end
t_chebfun3_construct = toc / N_c3 * 1000;

% ---- 3D: Evaluation of Chebfun3 at 500 points ----
f3 = chebfun3(@(x,y,z) cos(x+y+z));
xp3 = linspace(-1, 1, 500)';
yp3 = linspace(-1, 1, 500)';
zp3 = linspace(-1, 1, 500)';
warmup_val3 = f3(xp3, yp3, zp3);  % warm up
N_e3 = 50;
tic;
for i = 1:N_e3
    v3 = f3(xp3, yp3, zp3);
end
t_chebfun3_eval = toc / N_e3 * 1000;

% ---- Print results ----
fprintf('\n--- 1D Chebfun ---\n');
fprintf('construct sin:          %10.3f ms\n', t_construct);
fprintf('eval f(0.5):            %10.4f ms\n', t_eval);
fprintf('eval f(1000 pts):       %10.3f ms\n', t_eval1000);
fprintf('diff:                   %10.3f ms\n', t_diff);
fprintf('sum:                    %10.4f ms\n', t_sum);
fprintf('roots (sin(5*pi*x)):    %10.3f ms\n', t_roots);

fprintf('\n--- 1D Scaling (fixed-degree construction) ---\n');
for di = 1:length(degrees)
    fprintf('construct (trunc n=%5d): %10.3f ms\n', degrees(di), t_construct_n(di));
end

fprintf('\n--- 2D Chebfun2 ---\n');
fprintf('construct cos(x+y):     %10.3f ms\n', t_chebfun2_construct);
fprintf('eval (1000 pts):        %10.3f ms\n', t_chebfun2_eval);
fprintf('diff (x-direction):     %10.3f ms\n', t_chebfun2_diff);
fprintf('sum2:                   %10.4f ms\n', t_chebfun2_sum2);
fprintf('norm:                   %10.3f ms\n', t_chebfun2_norm);

fprintf('\n--- 3D Chebfun3 ---\n');
fprintf('construct cos(x+y+z):   %10.3f ms\n', t_chebfun3_construct);
fprintf('eval (500 pts):         %10.3f ms\n', t_chebfun3_eval);

% Write results to JSON for machine-readable import
fid = fopen('/scratch/gpfs/GILLES/mg6942/jaxchebfun/benchmarks/matlab_results.json', 'w');
fprintf(fid, '{\n');
fprintf(fid, '  "construct_sin_ms": %.6f,\n', t_construct);
fprintf(fid, '  "eval_single_ms": %.8f,\n', t_eval);
fprintf(fid, '  "eval_1000pts_ms": %.6f,\n', t_eval1000);
fprintf(fid, '  "diff_ms": %.6f,\n', t_diff);
fprintf(fid, '  "sum_ms": %.8f,\n', t_sum);
fprintf(fid, '  "roots_ms": %.6f,\n', t_roots);
fprintf(fid, '  "construct_n10_ms": %.6f,\n', t_construct_n(1));
fprintf(fid, '  "construct_n100_ms": %.6f,\n', t_construct_n(2));
fprintf(fid, '  "construct_n1000_ms": %.6f,\n', t_construct_n(3));
fprintf(fid, '  "chebfun2_construct_ms": %.6f,\n', t_chebfun2_construct);
fprintf(fid, '  "chebfun2_eval_ms": %.6f,\n', t_chebfun2_eval);
fprintf(fid, '  "chebfun2_diff_ms": %.6f,\n', t_chebfun2_diff);
fprintf(fid, '  "chebfun2_sum2_ms": %.8f,\n', t_chebfun2_sum2);
fprintf(fid, '  "chebfun2_norm_ms": %.6f,\n', t_chebfun2_norm);
fprintf(fid, '  "chebfun3_construct_ms": %.6f,\n', t_chebfun3_construct);
fprintf(fid, '  "chebfun3_eval_ms": %.6f\n', t_chebfun3_eval);
fprintf(fid, '}\n');
fclose(fid);
fprintf('\nResults written to /scratch/gpfs/GILLES/mg6942/jaxchebfun/benchmarks/matlab_results.json\n');
