function likelihood = CT_GetLikelihood( Y, mean1, mean0, sig1, sig0 )
%%
%
%
%
%% 
particle_num = size(Y,2);

mean1 = repmat(mean1,1,particle_num);
mean0 = repmat(mean0,1,particle_num);
sig1 = repmat(sig1,1,particle_num);
sig0 = repmat(sig0,1,particle_num);

%---���ո�˹�ܶȺ�����ÿһ����ѡĿ��Ĺ۲���Ȼֵ
p1 = (1./(sig1+eps) ) .* exp( (-1./(2*sig1.^2+eps) ).*(Y-mean1).^2 );
p0 = (1./(sig0+eps) ) .* exp( (-1./(2*sig0.^2+eps) ).*(Y-mean0).^2 );

r = log(p1./p0+eps);%+1��Ϊ�˷�ֹ���ָ�����
likelihood = sum(r);%��Ӧÿ����������Ȼֵ��������
%likelihood = likelihood/sum(likelihood);
end

