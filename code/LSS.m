function [dist, coef, error] = LSS( y, A, P )
% y :去中心化后的候选目标
% A :模板集
%dist : y到A的“距离”
%error: 误差向量
%%
lssparam.lambda = 0.1;    %求误差项时用到
lssparam.maxLoopNum = 20; %求系数时的最大迭代次数
lssparam.stopThr = 0.001; %求系数时，迭代终止条件

coef = zeros( size(A,2),1 );%初始化系数
error = zeros( size(y) );   %初始化误差向量
temp_error = zeros( size(y) );
dist_old = 0;%更新前的距离
dist_new = 0;%更新后的距离

for i=1:lssparam.maxLoopNum
    coef = P*(y-error);
    
    temp_error = y-A*coef;
    error = max( abs(temp_error)-lssparam.lambda, 0 ).*sign(temp_error);
    
    dist_new = 0.5*sum( (temp_error-error).^2 ) + lssparam.lambda*sum( abs(error) );
    if abs( dist_new - dist_old ) < lssparam.stopThr
        break;
    end
    dist_old = dist_new;
end

dist = dist_new;
end

