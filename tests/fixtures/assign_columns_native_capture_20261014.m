% Literal native test_assignColumns clauses 1--13 for both loop kinds.
% Source: tests/chebfun/test_assignColumns.m, Chebfun pin 7574c776.
root = fileparts(mfilename('fullpath'));
ref = '/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref';
addpath(genpath(ref));
format long g;
funlist = {@chebfun,@quasimatrix};
pref = chebfunpref();
for k = 1:2
    myfun = funlist{k};
    result = struct();
    result.k = k;
    result.myfun = func2str(myfun);
    result.pass = false(1,13);
    result.details = cell(1,13);
    result.prefClass = class(pref);
    result.prefChebfuneps = pref.chebfuneps;

    % Literal source seed and first test query draw.
    seedRNG(6178);
    result.rngBeforeX10 = rng;
    x = 2 * rand(10,1) - 1;
    result.x10 = x;
    result.rngAfterX10 = rng;

    f = myfun(@(x) [sin(x) cos(x) exp(x)],[-1 0 1],pref);
    g = myfun(@(x) x,[-1 1],pref);

    h = assignColumns(f,1,g);
    exact = @(x) [x cos(x) exp(x)];
    result.details{1} = capture_eval(h,x,exact,10,false);
    result.pass(1) = result.details{1}.passed;
    save(fullfile(root,sprintf('k%d_checkpoint.mat',k)),'result','-v7');

    h = assignColumns(f,3,g);
    exact = @(x) [sin(x) cos(x) x];
    result.details{2} = capture_eval(h,x,exact,10,false);
    result.pass(2) = result.details{2}.passed;
    save(fullfile(root,sprintf('k%d_checkpoint.mat',k)),'result','-v7');

    g = myfun(@(x) [x x.^2],[-1 0 1],pref);
    h = assignColumns(f,[3 1],g);
    exact = @(x) [x.^2 cos(x) x];
    result.details{3} = capture_eval(h,x,exact,10,false);
    result.pass(3) = result.details{3}.passed;
    save(fullfile(root,sprintf('k%d_checkpoint.mat',k)),'result','-v7');

    h = assignColumns(f,[2 2],g);
    exact = @(x) [sin(x) x.^2 exp(x)];
    result.details{4} = capture_eval(h,x,exact,10,false);
    result.pass(4) = result.details{4}.passed;
    save(fullfile(root,sprintf('k%d_checkpoint.mat',k)),'result','-v7');

    g = myfun(@(x) [x x.^2 x.^3],[-1 -0.5 0.5 1],pref);
    h = assignColumns(f,[1 2 3],g);
    exact = @(x) [x x.^2 x.^3];
    result.details{5} = capture_eval(h,x,exact,10,false);
    result.pass(5) = result.details{5}.passed;
    save(fullfile(root,sprintf('k%d_checkpoint.mat',k)),'result','-v7');

    hc = assignColumns(f,':',g);
    result.pass(6) = isequal(hc,h);
    result.details{6} = struct('equal',result.pass(6), ...
        'hcValues',feval(hc,x),'hValues',feval(h,x));
    save(fullfile(root,sprintf('k%d_checkpoint.mat',k)),'result','-v7');

    h = assignColumns(f,[2 1],[-0.5 0.5]);
    exact = @(x) [(0.5+0*x) (-0.5+0*x) exp(x)];
    result.details{7} = capture_eval(h,x,exact,10,false);
    result.pass(7) = result.details{7}.passed;
    save(fullfile(root,sprintf('k%d_checkpoint.mat',k)),'result','-v7');

    h = assignColumns(f.',[2 1],[-0.5;0.5]);
    exact = @(x) [(0.5+0*x) (-0.5+0*x) exp(x)].';
    result.details{8} = capture_eval(h,x,exact,10,false);
    result.pass(8) = result.details{8}.passed;
    save(fullfile(root,sprintf('k%d_checkpoint.mat',k)),'result','-v7');

    try
        assignColumns(f,[1 2 3],g.');
        result.pass(9) = false;
        result.details{9} = struct('errorIdentifier','','errorMessage','');
    catch ME
        result.pass(9) = strcmp(ME.identifier,'CHEBFUN:CHEBFUN:assignColumns:numCols');
        result.details{9} = struct('errorIdentifier',ME.identifier,'errorMessage',ME.message);
    end
    save(fullfile(root,sprintf('k%d_checkpoint.mat',k)),'result','-v7');

    try
        assignColumns(f,[1 2],g);
        result.pass(10) = false;
        result.details{10} = struct('errorIdentifier','','errorMessage','');
    catch ME
        result.pass(10) = strcmp(ME.identifier,'CHEBFUN:CHEBFUN:assignColumns:numCols');
        result.details{10} = struct('errorIdentifier',ME.identifier,'errorMessage',ME.message);
    end
    save(fullfile(root,sprintf('k%d_checkpoint.mat',k)),'result','-v7');

    try
        g2 = myfun(@(x) [x x.^2 x.^3],[0 1],pref);
        assignColumns(f,[1 2 3],g2);
        result.pass(11) = false;
        result.details{11} = struct('errorIdentifier','','errorMessage','');
    catch ME
        result.pass(11) = strcmp(ME.identifier,'CHEBFUN:CHEBFUN:assignColumns:domain');
        result.details{11} = struct('errorIdentifier',ME.identifier,'errorMessage',ME.message);
    end
    save(fullfile(root,sprintf('k%d_checkpoint.mat',k)),'result','-v7');

    h = assignColumns(f,[1 2 4],g);
    exact = @(x) [x x.^2 exp(x) x.^3];
    result.details{12} = capture_eval(h,x,exact,10,false);
    result.pass(12) = result.details{12}.passed;
    save(fullfile(root,sprintf('k%d_checkpoint.mat',k)),'result','-v7');

    % Continue the exact source RNG stream after constructors/checks 1--12.
    dom = [-Inf -3*pi];
    domCheck = [-1e6 -3*pi];
    result.rngBeforeX100 = rng;
    x100 = diff(domCheck) * rand(100,1) + domCheck(1);
    result.x100 = x100;
    result.rngAfterX100 = rng;
    op = @(x) [exp(x) x.*exp(x) (1-exp(x))./x];
    opg = @(x) exp(-x.^2);
    oph = @(x) [exp(x) exp(-x.^2) (1-exp(x))./x];
    f = myfun(op,dom,pref);
    g = myfun(opg,dom,pref);
    h = assignColumns(f,2,g);
    result.details{13} = capture_eval(h,x100,oph,1e2,true);
    result.pass(13) = result.details{13}.passed;
    result.details{13}.domain = dom;
    result.details{13}.domCheck = domCheck;
    result.details{13}.vscale = vscale(h);
    result.details{13}.matrixInfNormDefinition = 'norm(err,inf), exact source expression';
    save(fullfile(root,sprintf('k%d_checkpoint.mat',k)),'result','-v7');
    fprintf('k=%d class=%s pass=%s p13_norm=%.17g bound=%.17g\n', ...
        k,result.myfun,mat2str(result.pass),result.details{13}.norm, ...
        result.details{13}.bound);
end

function rec = capture_eval(f,x,exact,boundMultiplier,matrixInf)
    got = feval(f,x);
    expected = exact(x);
    err = got - expected;
    if matrixInf
        errNorm = norm(err,inf);
    else
        errNorm = norm(err(:),inf);
    end
    scale = vscale(f);
    bound = boundMultiplier * scale * eps;
    rec = struct('query',x,'values',got,'exact',expected,'error',err, ...
        'norm',errNorm,'vscale',scale,'bound',bound,'passed',errNorm < bound);
end
