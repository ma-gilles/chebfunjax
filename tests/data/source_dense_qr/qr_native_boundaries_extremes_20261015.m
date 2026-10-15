result=struct('matlab_version',version,'cases',{{}});
for m=[2,7,19,20,21,32,33,129]
    k=(1:m)';base=[mod(k*17,31)-15,mod(k*23+7,37)-18]/16;
    for exponent=[0,-500,-1020,500,1010]
        A=base*pow2(exponent);[Q,R]=qr(A,0);
        result.cases{end+1}=struct('m',m,'n',2,'exponent',exponent,'kind','real','A',A,'Q',Q,'R',R);
    end
end
for dims={[21,3],[21,4],[33,7],[129,9]}
    shape=dims{1};m=shape(1);n=shape(2);[i,j]=ndgrid(1:m,1:n);
    A=(mod(i.*(11+j)+7*j,37)-18)/16;
    [Q,R]=qr(A,0);result.cases{end+1}=struct('m',m,'n',n,'exponent',0,'kind','real_wide','A',A,'Q',Q,'R',R);
end
save('capture.mat','result','-v7');fprintf('Captured %d boundary/extreme QR controls\n',numel(result.cases));
