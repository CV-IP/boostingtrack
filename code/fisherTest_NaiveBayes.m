%每一列都是一个样本.
%返回错误率.
%%
function [rate, isclassify]  = fisherTest_NaiveBayes( samples, samples_weight,  flags,mean1, var1,mean0, var0  )
try
    warning off;
%     disp('进入NaiveBayes分类程序');
%   mean1 1*216
    rate = 0;
    isclassify = ones(1, length(flags));
%     num1 = size(mean1,2);
%     num0 = size(mean0,2);
    num = size(samples,2);
    mean1 = repmat(mean1',1,num);
    sig1 = repmat(var1',1,num);
    mean0 = repmat(mean0',1,num);
    sig0 = repmat(var0',1,num);
    Y = samples;
   
    
    p1 = (1./(sig1+eps) ) .* exp( (-1./(2*sig1.^2+eps) ).*(Y-mean1).^2 );% Y216*222  
    p0 = (1./(sig0+eps) ) .* exp( (-1./(2*sig0.^2+eps) ).*(Y-mean0).^2 );

    %r = log(p1./p0+eps);%+1是为了防止出现负无穷
    r = log(p1+eps) - log(p0 + eps);
    value = sum(r);%对应每个样本的似然值，行向量
    
    
    
   
    b = value .* flags;
    c = b<0;
    rate = sum ( samples_weight(c) );
    isclassify( c ) = 0;  
catch
        disp(mean1);
        disp(Y);
        disp(mean0);
        
    end
end