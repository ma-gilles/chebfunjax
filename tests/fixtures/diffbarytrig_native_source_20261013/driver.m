restoredefaultpath;
addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref');
zj=[0.2;1.1;2.4;4.6]; fj=[1+0.3i;-0.7+0.2i;0.4-0.8i;1.2]; wj=[0.7;-1.2+0.1i;0.8;0.3-0.2i];
zz=[0.35 0.2 7.2;1.7 3.1 -0.4];
records={};
for order=1:3
 for form={'odd','even'}
  r=struct('zz',zz,'zj',zj,'fj',fj,'wj',wj,'order',order,'form',form{1});
  try; r.values=diffbarytrig(zz,zj,fj,wj,order,form{1}); catch err; r.error=err.identifier;r.message=err.message;end
  records{end+1}=r;
 end
end
for form={'Even','evening','ODDtail'}
 r=struct('zz',zz,'zj',zj,'fj',fj,'wj',wj,'order',1,'form',form{1});
 try;r.values=diffbarytrig(zz,zj,fj,wj,form{1});catch err;r.error=err.identifier;r.message=err.message;end
 records{end+1}=r;
end
for kind=1:3
 r=struct('zz',zz,'zj',zj,'fj',fj,'wj',wj,'order',1,'form','odd');
 if kind==1;r.zz=complex([0.35 NaN 0 0],[0 0 Inf -Inf]);end
 if kind==2;r.zj=[0;2*pi;2.4;4.6];r.zz=[0 0.35 2*pi];end
 if kind==3;r.order=0;end
 try;r.values=diffbarytrig(r.zz,r.zj,r.fj,r.wj,r.order,r.form);catch err;r.error=err.identifier;r.message=err.message;end
 records{end+1}=r;
end
save('capture.mat','records','-v7');fprintf('CAPTURED %d cases\n',numel(records));
