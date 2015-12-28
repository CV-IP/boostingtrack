function [weak_classifier, selected_weak_classifier] =MyAdaBoost(X, Y, R, T, features_patch)
%%
% X:��patch��ϵ������ά
% Y:��patch��ϵ������ά
% R:ѧϰ��������������
% T:ѡ���������������
% features_patch:
%%
try
X_size = size( X, 3);%��������
Y_size = size( Y, 3);%��������
N = X_size + Y_size; %����������

F = size( features_patch, 2);%ÿ��ѵ��F����������������ѡ��һ����õ�

samples_coeff = zeros(size(X, 1), size(X, 2), N);
samples_coeff(:, :, 1:X_size) = X;
samples_coeff(:, :, X_size+1:end) = Y;%���е�patchϵ��

samples_flag = -1 * ones(1, N);
samples_flag(1:X_size) = 1;%���������ı�ǩ

samples_weight = ones(1, N) / Y_size/2;
samples_weight(1:X_size) = 1/X_size/2;%����Ȩ��

weak_classifier = cell( 1, R );

%----------ѵ��R����������------------
for i=1:R
    disp(i)
    % ����.
    Nnum = N*2/3;
    sub_samples_indx = sample_method( samples_weight,   Nnum);    
    best_param = [];
    rets =[];
    error_rate = zeros(F, 1);
    
    for j=1:F %ѵ��F����������������ѡ����õ�һ��
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
    % ����������ģ���Ȩ�����ʣ��������йز�����
    best_param.alpha = alpha;
    
    weak_classifier{ i } = best_param;   
    
    % ����������Ȩ��.
    if beta < 0.5
        beta = 0.5; 
    end
    for j = 1:N
        
        samples_weight( j ) = samples_weight( j ) * beta^(best_param.isclassify(j));    
    end
    
    % Ȩ�ع�һ��
    samples_weight = samples_weight / sum( samples_weight );    
end

rate = zeros(1, R);
% ѡ��T��������
for i = 1:R
    rate(i) = weak_classifier{i}.weight_error_rate;
end
[~, rate_indx] = sort( rate );
selected_weak_indx = rate_indx(1:T);

%���ˣ��Ѿ����ǿ��������ѡ��
selected_weak_classifier = weak_classifier( selected_weak_indx );
%��һ����Ȩ�أ�

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