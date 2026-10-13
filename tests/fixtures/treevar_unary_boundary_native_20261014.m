addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref');
names=strsplit('abs acos acosd acot acoth acsc acscd acsch airy asec asecd asech asin asind asinh atan atand atanh conj cos cosd cosh cot cotd coth csc cscd csch exp expm1 imag log log10 log1p log2 pow2 real sec secd sech sin sind sinh sqrt tan tand tanh uminus uplus');
nz=hex2num('8000000000000000');
values={-Inf,-2,-1,nz,0,1,2,Inf,NaN,.4,1i,-1i,.4+.3i,-.4+.3i};
labels={'real_ninf','real_n2','real_n1','real_n0','real_p0','real_p1','real_p2','real_pinf','real_nan','real_point4','imag_p1','imag_n1','regular_pp','regular_np'};
for a=[-2,-1,0,1,2]
 for b=[2^-40,-2^-40]
  values{end+1}=complex(a,b);labels{end+1}=sprintf('lip_%g_%+.0e',a,b);
 end
 values{end+1}=complex(a,0);labels{end+1}=sprintf('complex_%g_pzero',a);
 values{end+1}=complex(a,nz);labels{end+1}=sprintf('complex_%g_nzero',a);
end
values{end+1}=complex(Inf,1);labels{end+1}='complex_inf_1';
values{end+1}=complex(1,Inf);labels{end+1}='complex_1_inf';
inputs=cell(size(values));
for j=1:numel(values)
 x=values{j};inputs{j}=struct('label',labels{j},'real_hex',num2hex(real(x)),'imag_hex',num2hex(imag(x)),'isreal',isreal(x),'class',class(x));
end
records=cell(numel(names),numel(values));origins=cell(size(names));
record=struct('release',version('-release'),'version',version,'matlabroot',matlabroot,'names',{names},'inputs',{inputs},'records',{records},'origins',{origins});
for k=1:numel(names)
 fn=str2func(names{k});origins{k}=which(names{k});
 for j=1:numel(values)
  try
   y=fn(values{j});
   records{k,j}=struct('ok',true,'real_hex',num2hex(real(y)),'imag_hex',num2hex(imag(y)),'isreal',isreal(y),'class',class(y),'shape',size(y));
  catch err
   records{k,j}=struct('ok',false,'identifier',err.identifier,'message',err.message);
  end
 end
 record.records=records;record.origins=origins;record.completed_methods=k;
 save('checkpoint.mat','record','-v7');
end
save('capture.mat','record','-v7');
fid=fopen('capture.json','w');fprintf(fid,'%s\n',jsonencode(record));fclose(fid);
fprintf('Captured %d primitives x %d scalar inputs.\n',numel(names),numel(values));
