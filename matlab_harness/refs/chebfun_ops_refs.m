%% MATLAB references consumed by TestMATLABGoldenOps.
% Run with Chebfun 7574c77680d7e82b79626300bf255498271a72df on the path.
% This generator does not provide reference values until MATLAB executes it.
% New capture only: preserve any existing fixture for explicit review.

expected_commit = '7574c77680d7e82b79626300bf255498271a72df';
chebfun_source = which('chebfun');
assert(~isempty(chebfun_source), 'Chebfun must be on the MATLAB path.');
chebfun_root = fileparts(fileparts(chebfun_source));
assert(isempty(regexp(chebfun_root, '["$`\n\r]', 'once')), ...
    'Unsupported characters in Chebfun source path.');
[git_status, git_commit] = system(sprintf( ...
    'git -C "%s" rev-parse HEAD', chebfun_root));
assert(git_status == 0 && strcmp(strtrim(git_commit), expected_commit), ...
    'Reference capture requires the pinned Chebfun commit.');
[git_status, git_changes] = system(sprintf( ...
    'git -C "%s" status --porcelain --untracked-files=no', chebfun_root));
assert(git_status == 0 && isempty(strtrim(git_changes)), ...
    'Reference capture requires unchanged tracked Chebfun sources.');

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end
outfile = fullfile(outdir, 'chebfun_ops.mat');
assert(~exist(outfile, 'file'), 'Preserve the existing fixture for review.');

ref = struct();
f = chebfun(@sin, [-1, 1]);
g = chebfun(@cos, [-1, 1]);
ref.sin_integral = sum(f);
ref.cos_integral = sum(g);
% Native chebcoeffs returns T_0 first, matching the Python consumer.
ref.sin_diff_coeffs = chebcoeffs(diff(f));
ref.sin_roots = roots(f);
ref.sin_norm2 = norm(f, 2);
ref.capture_provenance = struct('chebfun_commit', expected_commit, ...
    'chebfun_source', chebfun_source, 'matlab_version', version, ...
    'matlab_release', version('-release'), 'generated_at', datestr(now, 30));
save(outfile, '-struct', 'ref', '-v7');
fprintf('Wrote %s from Chebfun %s.\n', outfile, expected_commit);
