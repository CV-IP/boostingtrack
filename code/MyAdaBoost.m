function [weak_classifier, selected_weak_classifier] =MyAdaBoost(X, Y, R, T, features_patch)
%%
% X:正patch块系数，三维
% Y:负patch块系数，三维
% R:学习的弱分类器个数
% T:选择的弱分类器个数
% features_patch:
%%
try
X_size = size( X, 3);%正样本数
Y_size = size( Y, 3);%负样本数
N = X_size + Y_size; %总体样本数

F = size( features_patch, 2);%每次训练F个弱分类器，从中选择一个最好的

samples_coeff = zeros(size(X, 1), size(X, 2), N);
samples_coeff(:, :, 1:X_size) = X;
samples_coeff(:, :, X_size+1:end) = Y;%所有的patch系数

samples_flag = -1 * ones(1, N);
samples_flag(1:X_size) = 1;%所有样本的标签

samples_weight = ones(1, N) / Y_size/2;
samples_weight(1:X_size) = 1/X_size/2;%样本权重

weak_classifier = cell( 1, R );

%----------训练R个弱分类器------------
for i=1:R
    disp(i)
    % 采样.
    Nnum = N*2/3;
    sub_samples_indx = sample_method( samples_weight,   Nnum);    
    best_param = [];
    rets =[];
    error_rate = zeros(F, 1);
    
    for j=1:F %训练F个弱分类器，从中选择最好的一个
        ret  = learnWeakClassifier( samples_weight , sub_samples_indx, j, features_patch, samples_coeff, samples_flag);
        rets = [rets; ret];
        error_rate(j) = ret.weight_error_rate;
        
    end
    min_error_rate = min( error_rate );
    min_indx = find( error_rate == min_error_rate );
    selected_indx = randi( [1 length(min_indx)], 1);
    best_param = rets( min_indx(selected_indx) );   
    
    beta = best_param.weight_error_rate / (( 1-best_param.weight_error_rate ) + 1e-6);
    alpha = log( 1/ (beta + 1e-6) );
    % 保存分类器的：加权错误率，分类器有关参数。
    best_param.alpha = alpha;
    
    weak_classifier{ i } = best_param;   
    
    % 更新样本的权重.
    if beta < 0.5
        beta = 0.5; 
    end
    for j = 1:N
        
        samples_weight( j ) = samples_weight( j ) * beta^(best_param.isclassify(j));    
    end
    
    % 权重归一化
    samples_weight = samples_weight / sum( samples_weight );    
end

rate = zeros(1, R);
% 选择T个分类器
for i = 1:R
    rate(i) = weak_classifier{i}.weight_error_rate;
end
[~, rate_indx] = sort( rate );
selected_weak_indx = rate_indx(1:T);

%至此，已经完成强分类器的选择。
selected_weak_classifier = weak_classifier( selected_weak_indx );
%归一化其权重：

weight_sum = 0;
for tmp_i = 1:T
    weight_sum = weight_sum + selected_weak_classifier{tmp_i}.alpha;
end
for tmp_i = 1:T
   selected_weak_classifier{tmp_i}.alpha = selected_weak_classifier{tmp_i}.alpha / weight_sum;
end







catch
        disp(samples_weight);
        disp(N);
        
        
    end

end