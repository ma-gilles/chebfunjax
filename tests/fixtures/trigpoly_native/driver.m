restoredefaultpath;
addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref');
degrees={1,2,[-1 0 2],[-1;0;2],1,[]};
domains={[-1 1],[-1 1],[-1 1],[-1 1],[2 5],[-1 1]};
records=cell(1,numel(degrees));
for k=1:numel(degrees)
 try
  f=trigpoly(degrees{k},domains{k}); x=linspace(domains{k}(1),domains{k}(end),7);
  record=struct('degrees',degrees{k},'domain',domains{k},'isempty',isempty(f),'query',x,'values',f(x));
  if ~isempty(f)
   record.coeffs=f.funs{1}.onefun.coeffs;record.transposed=f.isTransposed;record.pointValues=f.pointValues;
  end
  records{k}=record;
 catch err
  records{k}=struct('degrees',degrees{k},'error',err.identifier,'message',err.message);
 end
end
save('capture.mat','records','-v7');
fprintf('CAPTURED %d cases\n',numel(records));
