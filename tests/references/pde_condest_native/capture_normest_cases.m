function capture_normest_cases(outdir, pdeJ, pdeBefore)
% Native real condition-estimator observations, no fitted acceptance bounds.
assert(~isfolder(outdir), 'Capture:Exists', 'Output already exists');
assert(strcmp(which('condest'), fullfile(matlabroot,'toolbox','matlab','matfun','condest.m')));
assert(strcmp(which('normest1'), fullfile(matlabroot,'toolbox','matlab','matfun','normest1.m')));
mkdir(outdir);
matrices={pdeJ, reshape(sin((1:36)*1.234),6,6)+diag(3+(1:6)/5), ...
    eye(6)-(.999/6)*ones(6), eye(10), hilb(6), ...
    diag([ones(1,5),eps*6*(1+2^-20)]), ...
    diag([ones(1,5),eps*6*(1-2^-20)]), zeros(6), ...
    [4 1 -2 0; 0 3 1 -1; 2 0 5 1; 1 -2 0 4]};
names={'reachable_pde6','dense6','positive_inverse6','ties10','hilbert6', ...
    'index_below6','index_above6','singular6','small4'};
if isempty(pdeJ),matrices(1)=[];names(1)=[];end
records=cell(size(names));
L=[];U=[];trace={};
for k=1:numel(names)
    A=matrices{k}; n=size(A,1);
    if strcmp(names{k},'reachable_pde6'), rng(pdeBefore); else, rng(0,'twister'); end
    before=rng;
    c=condest(A); after=rng;
    [L,U,~]=lu(A,'vector');
    trace={}; est=[];v=[];w=[];iter=[]; normafter=[];
    if ~any(diag(U)==0)
        rng(before);
        [est,v,w,iter]=normest1(@action);
        normafter=rng;
        assert(isequaln(c,est*norm(A,1)), 'Capture:CondestValue', 'Operator replay differs');
        assert(isequaln(after,normafter), 'Capture:CondestRNG', 'Operator replay RNG differs');
    end
    name=names{k}; native_version=version;
    threshold=eps*nnz(A)*c; index_gt_one=(nnz(A)==0) || threshold>1;
    save(fullfile(outdir,[name '.mat']),'A','L','U','c','est','v','w','iter', ...
        'before','after','trace','name','native_version','threshold','index_gt_one','-v7');
    records{k}=struct('name',name,'n',n,'condition',c,'threshold',threshold, ...
        'index_gt_one',index_gt_one,'operator_calls',numel(trace),'iterations',iter);
end
% Consecutive calls advance one source stream rather than reseeding.
A=eye(6)-(.999/6)*ones(6); rng(42,'twister'); before=rng;
c1=condest(A); middle=rng; c2=condest(A); after=rng;
save(fullfile(outdir,'consecutive.mat'),'A','c1','c2','before','middle','after','-v7');
% Block-size and zero-operator cases exercise the estimator independently.
A=reshape(sin((1:36)*1.234),6,6)+diag(3+(1:6)/5); [L,U,~]=lu(A,'vector');
for t=[1 3 6]
    rng(0,'twister');before=rng;trace={};
    [est,v,w,iter]=normest1(@action,t);after=rng;
    save(fullfile(outdir,sprintf('block%d.mat',t)),'A','L','U','t','before','after', ...
        'est','v','w','iter','trace','-v7');
end
L=eye(6);U=eye(6);trace={}; rng(0,'twister');before=rng;
[est,v,w,iter]=normest1(@zeroaction);after=rng;
save(fullfile(outdir,'zero_operator6.mat'),'before','after','est','v','w','iter','trace','-v7');
f=fopen(fullfile(outdir,'summary.json'),'w');cleanup=onCleanup(@()fclose(f));
fprintf(f,'%s\n',jsonencode(struct('complete',true,'cases',{records},'version',version)));

    function value=action(flag,X)
        if strcmp(flag,'dim'),value=size(L,1);return;end
        if strcmp(flag,'real'),value=true;return;end
        if strcmp(flag,'notransp'),value=U\(L\X);
        elseif strcmp(flag,'transp'),value=L'\(U'\X);
        else,error('Capture:Flag','Unknown operator flag');end
        trace{end+1}=struct('flag',flag,'input',X,'output',value);
    end
    function value=zeroaction(flag,X)
        if strcmp(flag,'dim'),value=6;return;end
        if strcmp(flag,'real'),value=true;return;end
        value=zeros(size(X));
        trace{end+1}=struct('flag',flag,'input',X,'output',value);
    end
end
