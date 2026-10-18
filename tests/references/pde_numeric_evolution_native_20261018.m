function capture_numeric_evolution
ref='/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref';addpath(ref);
assert(strcmp(which('pde15s'),fullfile(ref,'@chebfun','pde15s.m')));
assert(strcmp(version('-release'),'2025b'));
records={};domains={[-1 1],[2 6]};
for d=1:2
 domain=domains{d};a=domain(1);b=domain(2);x=chebfun('x',domain);
 ell=2-5*(x-a)/(b-a);p=(x-a).*(b-x);
 for decay=[false true]
  current=struct('domain',domain,'decay',decay,'times',[0 .005 .01],'sample_x',a+(b-a)*[.125;.5;.875]);
  if decay
   u0=chebfun([ell+p;ell+2*p;ell+3*p]);
   fun=@(t,x,u,v,w)[.1*diff(u,2)-(x-a).*(b-x)*exp(-t)+.2*exp(-t);.2*diff(v,2)-4*(x-a).*(b-x)*exp(-2*t)+.8*exp(-2*t);.3*diff(w,2)-9*(x-a).*(b-x)*exp(-3*t)+1.8*exp(-3*t)];
  else
   u0=chebfun([ell;ell;ell]);fun=@(u,v,w)[.1*diff(u,2);.2*diff(v,2);.3*diff(w,2)];
  end
  bc=struct('left',2,'right',-3);opt=pdeset('N',9,'Plot','off','AdjustBCs',false,'RelTol',1e-9,'AbsTol',1e-11);
  try
   tic;[tt,solutions]=pde15s(fun,current.times,u0,bc,opt);current.wall_seconds=toc;
   current.returned_times=pack(tt);current.solution_size=size(solutions);
   % Native nonAdaptiveEvent never extends tOut (source lines133-134).
   assert(isequal(tt,0));assert(isequal(size(solutions),[3 3]));
   samples=zeros(3,3,3);analytic=zeros(3,3,3);xx=current.sample_x;
   for j=1:3
    for k=1:3
     samples(:,k,j)=feval(solutions{k,j},xx);
     analytic(:,k,j)=2-5*(xx-a)/(b-a)+double(decay)*k*(xx-a).*(b-xx)*exp(-k*current.times(j));
    end
   end
   current.samples=pack(samples);current.analytic=pack(analytic);current.returned_times=pack(tt);current.max_analytic_error=max(abs(samples(:)-analytic(:)));current.complete=true;
  catch err
   current.complete=false;current.error_id=err.identifier;current.error_message=err.message;
  end
  records{end+1}=current;
  result=struct('pin','7574c77680d7e82b79626300bf255498271a72df','matlab_version',version,'scope','Actual installed ode15s evolution via unmodified Chebfun pde15s; numeric nonzero scalar system endpoints; no observer override.','cases',{records});
  fid=fopen('/scratch/gpfs/CRYOEM/gilleslab/mg6942/_agent_scratch/pde_numeric_evolution_evidence_20261018/raw_native.json','w');assert(fid>=0);fwrite(fid,jsonencode(result),'char');fclose(fid);
  assert(current.complete,'Capture:CaseFailed',jsonencode(current));
 end
end
assert(numel(records)==4);disp('captured4actualnonzeronumericendpointtrajectories');
end
function out=pack(x)
out=struct('shape',size(x),'words',{cellstr(num2hex(x(:)))});
end
