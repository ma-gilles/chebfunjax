% fun_layer_refs.m — MATLAB references for the fun layer (Unbndfun /
% Singfun / Deltafun), the least-exercised Layer-3 classes. Everything is
% generated through the public MATLAB chebfun API (unbounded domains,
% 'exps' singular exponents, dirac); the Python tests exercise the classes
% directly since the chebfunjax factory does not yet route to them.
%
% Usage:
%   module load matlab/R2025b
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/fun_layer_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end
ref = struct();

% ---- Unbndfun: [0, inf) -------------------------------------------------
p1 = [0.5; 1.0; 2.0; 5.0; 10.0];
f = chebfun(@(x) exp(-x), [0 inf]);
ref.ub1_pts  = p1;
ref.ub1_eval = feval(f, p1);
ref.ub1_sum  = sum(f);                 % = 1
ref.ub1_dval = feval(diff(f), p1);     % = -exp(-x)

% ---- Unbndfun: (-inf, inf) ----------------------------------------------
p2 = [-5.0; -1.0; 0.0; 0.5; 3.0];
g = chebfun(@(x) 1./(1+x.^2), [-inf inf]);
ref.ub2_pts  = p2;
ref.ub2_eval = feval(g, p2);
ref.ub2_sum  = sum(g);                 % = pi

% ---- Singfun: inverse-sqrt endpoint singularities ------------------------
p3 = [-0.9; -0.5; 0.0; 0.4; 0.85];
s1 = chebfun(@(x) 1./sqrt(1-x.^2), 'exps', [-0.5 -0.5]);
ref.sg1_pts  = p3;
ref.sg1_eval = feval(s1, p3);
ref.sg1_sum  = sum(s1);                % = pi

% ---- Singfun: one-sided branch-point sqrt(1+x)*exp(x) --------------------
s2 = chebfun(@(x) sqrt(1+x).*exp(x), 'exps', [0.5 0]);
ref.sg2_pts  = p3;
ref.sg2_eval = feval(s2, p3);
ref.sg2_sum  = sum(s2);

% ---- Deltafun: dirac integration -----------------------------------------
x = chebfun('x');
d = dirac(x);
ref.dl1_sum = sum(d);                  % = 1
d2 = dirac(x - 0.3);
ref.dl2_sum  = sum(d2);                % = 1
h = cumsum(d);                          % heaviside
ref.dl_heavi_pts  = [-0.5; 0.5];
ref.dl_heavi_eval = feval(h, ref.dl_heavi_pts);

save(fullfile(outdir, 'fun_layer.mat'), '-struct', 'ref');
fprintf('fun_layer.mat written with %d fields\n', numel(fieldnames(ref)));
