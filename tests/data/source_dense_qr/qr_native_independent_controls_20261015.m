result=struct();
result.matlab_version=version;
result.cases={};
for n=[1,2,7,17,33,129,2049]
    k=(1:n)';
    base=[mod(k*17,31)-15,mod(k*23+7,37)-18]/16;
    for kind=1:5
        switch kind
            case 1, A=base;
            case 2, A=[base(:,1),base(:,1)];
            case 3, A=[zeros(n,1),base(:,2)];
            case 4, A=[base(:,1),zeros(n,1)];
            case 5, A=zeros(n,2);
        end
        [Q,R]=qr(A,0);
        entry=struct('m',n,'kind',kind,'A',A,'Q',Q,'R',R);
        result.cases{end+1}=entry;
    end
end
save('capture.mat','result','-v7');
fprintf('Captured %d independent economy QR controls\n',numel(result.cases));
