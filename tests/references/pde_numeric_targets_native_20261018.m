function capture_numeric_targets
ref='/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref';addpath(ref);
assert(strcmp(which('pdeSolve'),fullfile(ref,'@chebfun','pdeSolve.m')));
assert(strcmp(version('-release'),'2025b'));
records={};current=struct;
for domain={[-1 1],[2 6]}
 for targets={[0 0],[2 -3],[-3 2]}
  for adjust=[false true]
   current=struct('domain',domain{1},'targets',targets{1},'adjust',adjust);
   x=chebfun('x',domain{1});u0=chebfun([1+x+x.^2;2-2*x+2*x.^2;3+3*x+3*x.^2]);
   bc=struct('left',targets{1}(1),'right',targets{1}(2));
   opt=pdeset('N',9,'Plot','off','AdjustBCs',adjust,'ODESolver',@observe);
   try
    [~,unused]=pdeSolve(@(u,v,w)[.1*diff(u,2);.2*diff(v,2);.3*diff(w,2)],[0 .005 .01],u0,bc,opt); %#ok<ASGLU>
    error('Capture:MissingStop','Assembly observer was not reached');
   catch err
    assert(strcmp(err.identifier,'Capture:AssemblyComplete'),err.message);
   end
  end
 end
end
assert(numel(records)==12);
result=struct('pin','7574c77680d7e82b79626300bf255498271a72df','matlab_version',version,'scope','Actual unmodified pdeSolve assembly with numeric scalar system endpoints; observer stops before installed ODE solver. No evolution/native trajectory claim.','cases',{records});
fid=fopen('/scratch/gpfs/CRYOEM/gilleslab/mg6942/_agent_scratch/pde_numeric_target_evidence_20261018/raw_native.json','w');assert(fid>=0);fwrite(fid,jsonencode(result),'char');fclose(fid);disp('captured12actualnumericendpointassemblies');
 function [t,y]=observe(fun,span,initial,opt) %#ok<INUSD,STOUT>
  names={'B','P','M','x','n','BCVALOFFSET','numBCs','numLinBCs','numLBCs','numRBCs','q'};info=functions(fun);context=struct;
  for k=1:numel(info.workspace)
   ws=info.workspace{k};
   for j=1:numel(names)
    name=names{j};if isfield(ws,name)&&isnumeric(ws.(name)),context.(name)=ws.(name);end
   end
  end
  assert(all(isfield(context,names)));assert(context.n==9&&context.numBCs==6&&context.numLinBCs==6&&context.numLBCs==0&&context.numRBCs==0);
  assert(isequal(size(context.B),[6 27])&&isequal(size(context.P),[27 27]));
  expected=repelem(current.targets(:),3);assert(isequal(context.q,expected));
  xx=context.x(:);probe=[1+xx+2*xx.^2,2-3*xx+2*xx.^2+xx.^3,3+xx.^2-2*xx.^3];
  current.initial=pack(initial(:));current.B=pack(context.B);current.P=pack(context.P);current.M=pack(context.M);current.x=pack(context.x);current.q=pack(context.q);current.offset=pack(context.BCVALOFFSET);current.probe=pack(probe);current.initial_rhs=pack(fun(span(1),initial(:)));current.probe_rhs=pack(fun(.003,probe(:)));
  records{end+1}=current;error('Capture:AssemblyComplete','Stopped after actual assembly, before ODE solve');
 end
end
function out=pack(x)
out=struct('shape',size(x),'words',{cellstr(num2hex(x(:)))});
end
