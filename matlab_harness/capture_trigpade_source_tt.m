function capture_trigpade_source_tt(sourceRoot, outputFile)
% Capture the original unseeded test prefix, without substituting RNG inputs.
% Run only in an explicitly selected native MATLAB RNG/test-runner context.
% Writes primitive tt and RNG states; does not certify the remaining tests.
sourceFile = fullfile(sourceRoot, 'tests', 'chebfun', 'test_trigpade.m');
source = fileread(sourceFile);
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(uint8(source));
actualHash = lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
expectedHash = '3f6ae01eb993a95b46ee04cbfc61558c2d5d539083d9a1ceafcfaf0372fc1c76';
assert(strcmp(actualHash,expectedHash),'Pinned trigpade source hash mismatch');
initialState = rng;
source = strrep(source,'function pass = test_trigpade(pref)', ...
    'function pass = capture_trigpade_prefix(pref)');
needle = 'tt = -1+2*rand(100,1);';
assert(numel(strfind(source,needle)) == 1);
inject = [ 'rngBeforeTT = rng;' newline needle newline ...
    'captured.tt = tt; captured.rng_before_tt = rngBeforeTT;' newline ...
    'captured.rng_after_tt = rng;' newline ...
    'captured.source_commit = ''7574c77680d7e82b79626300bf255498271a72df'';' newline ...
    'captured.source_sha256 = ''3f6ae01eb993a95b46ee04cbfc61558c2d5d539083d9a1ceafcfaf0372fc1c76'';' newline ...
    'assignin(''caller'',''captured'',captured); return;' ];
source = strrep(source,needle,inject);
outputDirectory = fileparts(outputFile);
assert(~isempty(outputDirectory),'Use an absolute shared-scratch output path');
staging = tempname(outputDirectory); mkdir(staging);
filename = fullfile(staging,'capture_trigpade_prefix.m');
fid = fopen(filename,'w'); assert(fid>=0); fwrite(fid,source); fclose(fid);
addpath(staging,'-begin');
cleanup = onCleanup(@() remove_staging(staging));
capture_trigpade_prefix();
captured.rng_entry = initialState;
captured.capture_scope = 'Original test prefix through tt; remaining test not run';
fid = fopen(outputFile,'w'); assert(fid>=0);
fprintf(fid,'%s\n',jsonencode(captured)); fclose(fid);
end

function remove_staging(staging)
rmpath(staging);
delete(fullfile(staging,'capture_trigpade_prefix.m'));
rmdir(staging);
end
