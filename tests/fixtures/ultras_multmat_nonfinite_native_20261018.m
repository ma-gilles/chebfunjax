function capture_native
ref='/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref';addpath(ref);
inputs={};base=[.5;.25;.125;0;0;0];
for index=[1 4 6]
 for v=[NaN Inf -Inf]
  a=base;a(index)=v;inputs(end+1,:)={sprintf('real_index%d_value%g',index,v),6,2,a}; %#ok<AGROW>
 end
end
for parts={[Inf .25],[.125 Inf],[Inf 0],[0 Inf],[NaN .25],[.125 NaN],[Inf Inf],[Inf -Inf]}
 z=parts{1};a=complex(base,zeros(size(base)));a(4)=complex(z(1),z(2));
 inputs(end+1,:)={sprintf('complex_re%g_im%g',z(1),z(2)),6,2,a}; %#ok<AGROW>
end
for lambda=[2 3]
 a=zeros(8,1);a(1)=1;a(3)=-1.5*2^53;a(5)=-6*2^53;a(2)=NaN;
 inputs(end+1,:)={sprintf('finite_row_cancellation_lam%d',lambda),8,lambda,a}; %#ok<AGROW>
end
inputs(end+1,:)={'complex_finite_control',8,3,[.5+.25i;0;.125-.2i;0;0;.25+.125i;0;0]};
cases=cell(size(inputs,1),1);
for k=1:size(inputs,1)
 name=inputs{k,1};n=inputs{k,2};lambda=inputs{k,3};a=inputs{k,4};
 c=struct('name',name,'n',n,'lambda',lambda,'input_isreal',isreal(a),'input_re_hex',{cellstr(num2hex(real(a(:))))},'input_im_hex',{cellstr(num2hex(imag(a(:))))});
 S=ultraS.convertmat(n,0,lambda-1);[i,j,v]=find(S);
 c.conversion_nnz=nnz(S);c.conversion_nzmax=nzmax(S);c.conversion_is_sparse=issparse(S);
 c.conversion_i=i;c.conversion_j=j;c.conversion_v_hex=cellstr(num2hex(v));
 converted=S*a;c.converted_isreal=isreal(converted);c.converted_re_hex=cellstr(num2hex(real(converted)));c.converted_im_hex=cellstr(num2hex(imag(converted)));
 try
  M=ultraS.multmat(n,a,lambda);c.output_isreal=isreal(M);c.output_shape=size(M);
  c.output_re_hex=cellstr(num2hex(real(full(M(:)))));c.output_im_hex=cellstr(num2hex(imag(full(M(:)))));c.error_id='';c.error_message='';
 catch err
  c.error_id=err.identifier;c.error_message=err.message;
 end
 decisions=[];
 for nn=1:n-2
  tail=converted(nn+3:end);stop=~isempty(tail)&&all(abs(tail)<eps);decisions=[decisions stop]; %#ok<AGROW>
  if stop,break,end
 end
 c.source_derived_tail_decisions=decisions;cases{k}=c;
end
result=struct('source_pin','7574c77680d7e82b79626300bf255498271a72df','matlab_version',version,'which_multmat',which('ultraS.multmat'),'which_convertmat',which('ultraS.convertmat'),'input_format','Literal real/complex inputs; exact num2hex words; complex storage flag preserved; output matrix column-major. Sparse find returns CSC column order.','cases',{cases});
fid=fopen('/scratch/gpfs/CRYOEM/gilleslab/mg6942/_agent_scratch/ultras_nonfinite_native_20261018/raw_native.json','w');assert(fid>=0);fwrite(fid,jsonencode(result),'char');fclose(fid);disp('captured 20 unmodified native multmat/conversion cases');
end
