function rate = MyAdaboostClassify( features_patch, selected_weak_classifier, sample, sample_flag )
%% 对每个分类器，提取相应的特征，进行分类，并输出加权和.
%
%
%

%
%%
T = length(selected_weak_classifier);
rate = 0;
for tmp_i = 1:T
    swc = selected_weak_classifier{tmp_i};
    weight = swc.alpha;
    feature_indx = swc.feature_indx;
    %---------svm&logistic---------------------
%      w_opt = swc.w_opt;
%     w0 = swc.w0;
%     
%     sample_feature = get_sample_feature(feature_indx, features_patch, sample);        
%     value = w_opt' * sample_feature + w0;

    %-------NaiveBayes------------------------
    mean1 = swc.mean1;
     var1 = swc.var1;
      mean0 = swc.mean0;
       var0 = swc.var0;
    sample_feature = get_sample_feature(feature_indx, features_patch, sample); 
     mean1 = mean1';
    sig1 = var1';
    mean0 = mean0';
    sig0 = var0';
    Y =  get_sample_feature(feature_indx, features_patch, sample);  
    p1 = (1./(sig1+eps) ) .* exp( (-1./(2*sig1.^2+eps) ).*(Y-mean1).^2 );% Y216*222  
    p0 = (1./(sig0+eps) ) .* exp( (-1./(2*sig0.^2+eps) ).*(Y-mean0).^2 );
    %r = log(p1./p0+eps);%+1是为了防止出现负无穷
    r = log(p1+eps) - log(p0 + eps);
    value = sum(r);
    
    
    %判定与标签不一致.
%     if value * sample_flag < 0
%         rate = rate - weight;        
%     else
%         rate = rate + weight;
%     end  
    rate = rate + weight * value';
    
end

function f = get_sample_feature(feature_indx, features_patch, sample)
    sample = sample( :,features_patch(:,feature_indx) );
%     sample = sum( sample,2 )./size( sample,2 ); % average pooling
%     sample = max( sample,[],2 ); % max pooling
    sample  = reshape(sample,size( sample, 1)*size( sample, 2), size( sample, 3) );%concatenate pooling
    f = sample;
end


end