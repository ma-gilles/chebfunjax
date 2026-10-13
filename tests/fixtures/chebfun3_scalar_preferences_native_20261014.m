restoredefaultpath;
M = '/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref';
addpath(M);
assert(strcmp(which('chebfun'), fullfile(M, '@chebfun', 'chebfun.m')));
assert(strcmp(which('chebfun3'), fullfile(M, '@chebfun3', 'chebfun3.m')));
format long g;
out = struct();
out.sourcePin = '7574c77680d7e82b79626300bf255498271a72df';
out.matlabVersion = version;
out.whichChebfun = which('chebfun');
out.whichChebfun3 = which('chebfun3');
defaultPref = chebfunpref();
out.defaultPrefTech = func2str(defaultPref.tech);
dom = [-2 1 -2 1 -2 1];

f = chebfun3(3, 'trig', dom);
out.scalarTrigFlag = describe(f, dom, 3);

% The scalar branch recurses without the explicit pref argument. First make
% the session default Trig, then request a local C2 preference.
chebfunpref.setDefaults('tech', @trigtech);
f = chebfun3(3, 'tech', @chebtech2, dom);
out.scalarLocalC2UnderSessionTrig = describe(f, dom, 3);
chebfunpref.setDefaults('factory');
save(fullfile(fileparts(mfilename('fullpath')), 'capture.mat'), 'out', '-v7');

f = chebfun3(3, 'eps', 1e-8, 'trig', dom);
out.scalarExplicitEpsTrig = describe(f, dom, 3);
save(fullfile(fileparts(mfilename('fullpath')), 'capture.mat'), 'out', '-v7');

pref = chebfunpref();
pref.tech = @trigtech;
f = chebfun3(3, pref, dom);
out.scalarExplicitTrigPref = describe(f, dom, 3);

chebfunpref.setDefaults('tech', @trigtech);
f = chebfun3(3, pref, dom);
out.scalarExplicitTrigPrefUnderSessionTrig = describe(f, dom, 3);
chebfunpref.setDefaults('factory');
save(fullfile(fileparts(mfilename('fullpath')), 'capture.mat'), 'out', '-v7');

f = chebfun3(3, 'coeffs', 'trig', dom);
out.scalarCoeffsTrig = describe(f, dom, 3);
save(fullfile(fileparts(mfilename('fullpath')), 'capture.mat'), 'out', '-v7');

try
    f = chebfun3(3, 'equi', 'trig', dom);
    out.scalarEquiTrig = describe(f, dom, 3);
catch ME
    out.scalarEquiTrig = struct('errorIdentifier', ME.identifier, ...
        'errorMessage', ME.message);
end
save(fullfile(fileparts(mfilename('fullpath')), 'capture.mat'), 'out', '-v7');

fprintf('default session technology: %s\n', out.defaultPrefTech);
fprintf('scalar + trig: %s, rank=%s, length=%s\n', ...
    out.scalarTrigFlag.factorTech{1}, mat2str(out.scalarTrigFlag.rank), ...
    mat2str(out.scalarTrigFlag.length));
fprintf('scalar + eps + trig: %s, rank=%s, length=%s\n', ...
    out.scalarExplicitEpsTrig.factorTech{1}, ...
    mat2str(out.scalarExplicitEpsTrig.rank), ...
    mat2str(out.scalarExplicitEpsTrig.length));
fprintf('scalar + explicit Trig preference: %s\n', ...
    out.scalarExplicitTrigPref.factorTech{1});
fprintf('local C2 under Trig session: %s; explicit Trig pref under Trig session: %s\n', ...
    out.scalarLocalC2UnderSessionTrig.factorTech{1}, ...
    out.scalarExplicitTrigPrefUnderSessionTrig.factorTech{1});
fprintf('scalar coefficient + trig: %s\n', out.scalarCoeffsTrig.factorTech{1});
fprintf('scalar + equi + trig: %s\n', ...
    jsonencode(out.scalarEquiTrig));

function d = describe(f, dom, value)
    [rx, ry, rz] = rank(f);
    d = struct('class', class(f), ...
        'factorTech', {{class(f.cols.funs{1}.onefun), ...
                        class(f.rows.funs{1}.onefun), ...
                        class(f.tubes.funs{1}.onefun)}}, ...
        'rank', [rx ry rz], 'length', length(f), 'domain', f.domain, ...
        'query', [0.17 0.23 0.31], ...
        'queryValue', feval(f, 0.17, 0.23, 0.31), ...
        'inputValue', value, 'requestedDomain', dom, 'vscale', vscale(f));
end
