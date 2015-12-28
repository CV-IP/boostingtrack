function [mean1, mean0, sig1, sig0] = CT_UpdateClassifer(pos_samples, neg_samples, mean1, mean0, sig1, sig0, lambda)
%
%
%
%
%% 
%---求正样本的均值和方差
pos_num = size(pos_samples, 2);%正样本的数目
pos_mean = mean(pos_samples,2);%正样本的均值
pos_var = mean( (pos_samples-repmat(pos_mean,1,pos_num)).^2 ,2);%正样本的方差

%---求负样本的均值和方差
neg_num = size(neg_samples, 2);%负样本的数目
neg_mean = mean(neg_samples,2);%负样本的均值
neg_var = mean( (neg_samples-repmat(neg_mean,1,neg_num)).^2 ,2);%负样本的方差

%---更新正样本的均值和方差
mean1 = lambda*mean1 + (1-lambda)*pos_mean;
sig1 = sqrt( lambda*sig1.^2 + (1-lambda)*pos_var + lambda*(1-lambda)*(mean1-pos_mean).^2 );

%---更新负样本的均值和方差
mean0 = lambda*mean0 + (1-lambda)*neg_mean;
sig0 = sqrt( lambda*sig0.^2 + (1-lambda)*neg_var + lambda*(1-lambda)*(mean0-neg_mean).^2 );
end