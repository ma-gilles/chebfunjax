%% MATLAB reference data for additional Chebop (ODE/BVP/eig) operations.
%
% Extends the operators golden ref (which had 3 BVPs + 1 eig) with
% variable-coefficient, Neumann, and higher-order BVPs plus classic
% eigenvalue problems, cross-validating chebfunjax's Chebop against
% MATLAB @chebop at machine precision (or documented adaptive-size
% tolerance).  Added by Claude Opus 4.8.
%
% Usage:
%   module load matlab/R2025b
%   cd /home/mg6942/chebfunjax
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/chebop_extras_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();

pts = linspace(-0.9, 0.9, 13).';
ref.pts = pts;

% ---- 1. Variable-coefficient BVP: u'' + x u = 1, u(-1)=u(1)=0 ----
L1 = chebop(@(x,u) diff(u,2) + x.*u, [-1, 1]);
L1.lbc = 0; L1.rbc = 0;
u1 = L1 \ 1;
ref.varcoeff_vals = feval(u1, pts);

% ---- 2. Neumann/mixed BVP: u'' - u = 1, u'(-1)=0, u(1)=0 ----
L2 = chebop(@(x,u) diff(u,2) - u, [-1, 1]);
L2.lbc = @(u) diff(u);   % Neumann on left
L2.rbc = 0;              % Dirichlet on right
u2 = L2 \ 1;
ref.neumann_vals = feval(u2, pts);

% ---- 3. Variable-coefficient with rhs = cos(pi x) ----
L3 = chebop(@(x,u) diff(u,2) + sin(pi*x).*u, [-1, 1]);
L3.lbc = 0; L3.rbc = 0;
u3 = L3 \ chebfun(@(x) cos(pi*x), [-1, 1]);
ref.varcoeff2_vals = feval(u3, pts);

% ---- 4. Harmonic-oscillator eigenvalues: -u'' + x^2 u = lam u ----
%        on [-6, 6] with Dirichlet BCs.  Exact eigenvalues: 1,3,5,7,9,...
Lh = chebop(@(x,u) -diff(u,2) + x.^2.*u, [-6, 6]);
Lh.bc = 0;
ref.harmonic_eigs = sort(eigs(Lh, 5, 'sr'));

% ---- 5. Dirichlet Laplacian eigenvalues on [0, 1]: pi^2 n^2 ----
Ld = chebop(@(x,u) -diff(u,2), [0, 1]);
Ld.bc = 0;
ref.laplacian01_eigs = sort(eigs(Ld, 6, 'sr'));

save(fullfile(outdir, 'chebop_extras.mat'), '-struct', 'ref');
fprintf('Wrote chebop_extras.mat with %d fields.\n', numel(fieldnames(ref)));
