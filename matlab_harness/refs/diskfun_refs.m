% diskfun.m — MATLAB references for the diskfun module.
% chebfunjax Diskfun is polar-native (f(theta,r), theta in [-pi,pi], r in [0,1]),
% so the MATLAB references are built in 'polar' mode to compare like-for-like.
% Pins evaluation, sum2 (integral over the unit disk), and numerical rank at
% rtol 1e-12.
%
% Usage:
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/diskfun.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end
ref = struct();

% Polar evaluation points (theta in [-pi,pi], r in [0,1)).
tpts = [-2.5; -1.0;  0.0;  1.2;  2.8; -0.5];
rpts = [ 0.2;  0.5;  0.7;  0.3;  0.9;  0.6];
ref.tpts = tpts;
ref.rpts = rpts;

pfuns = { @(t,r) r.^2, ...
          @(t,r) r.*cos(t), ...
          @(t,r) exp(r.*cos(t)), ...
          @(t,r) 1 + (r.*cos(t)).*(r.*sin(t)) };
for i = 1:numel(pfuns)
    f = diskfun(pfuns{i}, 'polar');
    ref.(sprintf('f%d_eval', i)) = feval(f, tpts, rpts, 'polar');
    ref.(sprintf('f%d_sum', i))  = sum2(f);
    ref.(sprintf('f%d_rank', i)) = rank(f);
end

save(fullfile(outdir, 'diskfun.mat'), '-struct', 'ref');
fprintf('diskfun.mat: %d fields\n', numel(fieldnames(ref)));
