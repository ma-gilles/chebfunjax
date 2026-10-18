function capture_native
ref='/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref';addpath(ref);
e=eps;
inputs={
 'disk_r2',9,[.5;0;.5],2;
 'zero_gap_degree5',9,[.5;0;.125;0;0;.25],2;
 'complex_degree5',9,[.5+.25i;0;.125-.2i;0;0;.25+.125i],3;
 'below_eps',6,[.5;.25;.125;4*e;0;0],2;
 'at_eps',6,[.5;.25;.125;8*e;0;0],2;
 'complex_below_eps',6,[.5;.25;.125;(4+4i)*e;0;0],2;
 'complex_above_eps',6,[.5;.25;.125;(6+6i)*e;0;0],2;
 'empty_final_tail',3,[.5;.25;.125],2;
 'n1_scalar_complex',1,.5+.25i,2;
 'n2_complex',2,[.5;.25i],2;
 'nan_tail',6,[.5;.25;.125;NaN;0;0],2;
 'inf_tail',6,[.5;.25;.125;Inf;0;0],2};
cases=cell(size(inputs,1),1);
for k=1:size(inputs,1)
 name=inputs{k,1};n=inputs{k,2};a=inputs{k,3};lambda=inputs{k,4};
 c=struct('name',name,'n',n,'lambda',lambda,'input_re_hex',{cellstr(num2hex(real(a(:))))},'input_im_hex',{cellstr(num2hex(imag(a(:))))});
 try
  M=ultraS.multmat(n,a,lambda);
  c.output_shape=size(M);c.output_re_hex=cellstr(num2hex(real(full(M(:)))));c.output_im_hex=cellstr(num2hex(imag(full(M(:)))));
  c.error_id='';c.error_message='';
 catch err
  c.error_id=err.identifier;c.error_message=err.message;
 end
 % Capture normalization separately; native multiplication remains unmodified.
 a=a(:);a=[a;zeros(max(0,n-length(a)),1)];a=a(1:n);
 if n>1
  converted=ultraS.convertmat(n,0,lambda-1)*a;
  c.converted_re_hex=cellstr(num2hex(real(converted)));
  c.converted_im_hex=cellstr(num2hex(imag(converted)));
  decisions=[];
  for nn=1:n-2
   tail=converted(nn+3:end);
   stop=~isempty(tail)&&all(abs(tail)<eps);
   decisions=[decisions stop]; %#ok<AGROW>
   if stop,break,end
  end
  c.tail_decisions=decisions;
 else
  c.tail_decisions=[];
 end
 cases{k}=c;
end
result=struct('source_pin','7574c77680d7e82b79626300bf255498271a72df','matlab_version',version,'which_multmat',which('ultraS.multmat'),'input_format','Literal source inputs; num2hex exact binary64 words; complex parts captured separately. Column-major matrix serialization.','cases',{cases},'if_empty_condition',false);
output='/scratch/gpfs/CRYOEM/gilleslab/mg6942/_agent_scratch/ultras_tail_native_20261018/raw_native.json';
fid=fopen(output,'w');assert(fid>=0);fwrite(fid,jsonencode(result),'char');fclose(fid);
disp('captured 12 unmodified native multmat controls');
end
