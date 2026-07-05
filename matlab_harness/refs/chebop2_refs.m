%% Second batch of Chebop reference data: variable-coeff eig + BVPs.
%
% Cross-validates more @chebop behaviours vs MATLAB R2025b: a Mathieu
% eigenvalue problem, an Airy-type oscillatory BVP, and a
% cosine-coefficient BVP.  Added by Claude Opus 4.8.
%
% Usage:
%   module load matlab/R2025b
%   cd /home/mg6942/chebfunjax
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/chebop2_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();

% ---- Mathieu eigenvalues: -u'' + 4 cos(2x) u = lam u on [0,pi], Dirichlet
Lm = chebop(@(x,u) -diff(u,2) + 4*cos(2*x).*u, [0, pi]);
Lm.bc = 0;
ref.mathieu_eigs = sort(eigs(Lm, 4, 'sr'));

% ---- Airy-type oscillatory BVP: u'' - 30 x u = 0, u(-1)=1, u(1)=0 ----
pts1 = linspace(-0.9, 0.9, 13).';
ref.pts1 = pts1;
La = chebop(@(x,u) diff(u,2) - 30*x.*u, [-1, 1]);
La.lbc = 1; La.rbc = 0;
ua = La \ 0;
ref.airy_vals = feval(ua, pts1);

% ---- cosine-coefficient BVP: u'' + cos(pi x) u = 1, Dirichlet ----
Lc = chebop(@(x,u) diff(u,2) + cos(pi*x).*u, [-1, 1]);
Lc.lbc = 0; Lc.rbc = 0;
uc = Lc \ 1;
ref.cosbvp_vals = feval(uc, pts1);

save(fullfile(outdir, 'chebop2.mat'), '-struct', 'ref');
fprintf('Wrote chebop2.mat with %d fields.\n', numel(fieldnames(ref)));
