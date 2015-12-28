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

%---按照高斯密度函数求每一个候选目标的观测似然值
p1 = (1./(sig1+eps) ) .* exp( (-1./(2*sig1.^2+eps) ).*(Y-mean1).^2 );
p0 = (1./(sig0+eps) ) .* exp( (-1./(2*sig0.^2+eps) ).*(Y-mean0).^2 );

r = log(p1./p0+eps);%+1是为了防止出现负无穷
likelihood = sum(r);%对应每个样本的似然值，行向量
%likelihood = likelihood/sum(likelihood);
end

