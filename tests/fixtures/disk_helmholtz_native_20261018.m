function capture_native
addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref');
addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref/tests/diskfun');
result.original_pass = test_helmholtz();
m=17; n=12;
D1=ultraS.diffmat(m,1); D2=ultraS.diffmat(m,2);
Mr=ultraS.multmat(m,[0;1],1); Mr2=ultraS.multmat(m,[.5;0;.5],2);
Mr2c=ultraS.multmat(m,[.5;0;.5],0);
S1=ultraS.convertmat(m,0,1); S12=ultraS.convertmat(m,1,1);
F=reshape(mod(0:n*m-1,11)-5,n,m)/16;
bc=(mod((0:n-1)',5)-2)/8;
Ks=[0 .7 2i .7+2e-15i];
for c=1:numel(Ks)
 K=Ks(c); L=Mr2*D2+S12*Mr*D1+K^2*S1*Mr2c; L=L(1:end-2,:); S=S1(1:end-2,:);
 CFS=zeros(m,n); direct=CFS;
 for k=1:n
  j=-n/2+k-1; a=mod(j,2)+1; B=L-j^2*S;
  kB=[zeros(1,(m-1)/2+abs(mod(j,2)-1)); B(a:2:end,a:2:end)]; kB(1,1)=1;
  w=zeros(size(kB,1),1);w(1)=1;
  b=[bc(k);F(k,a:2:end-2).'];
  ib=kB\b;iw=kB\w;
  CFS(a:2:end,k)=ib-sum(ib(2:end))*iw/(1+sum(iw(2:end)));
  kB(1,:)=1;direct(a:2:end,k)=kB\b;
 end
 item.K=[real(K),imag(K)]; item.operator_re=full(real(L));item.operator_im=full(imag(L));
 item.conversion=full(S);item.forcing=F;item.boundary=bc;
 item.coefficients_re=real(CFS);item.coefficients_im=imag(CFS);
 item.direct_error=max(abs(CFS(:)-direct(:)));
 result.cases{c}=item;
end
fid=fopen('/scratch/gpfs/CRYOEM/gilleslab/mg6942/_agent_scratch/disk_helmholtz_evidence_20261018/native.json','w');fprintf(fid,'%s',jsonencode(result));fclose(fid);
disp(result.original_pass);
end
