%% 检查采集的负样本是否合适，把不合适的标记出来
%patch_dic: 规范化后的字典
%neg_samples：采集到的负样本

neg_patch0 = GetPatch( neg_samples, patch_size, patch_step );
neg_patch0 = reshape( neg_patch0, size(neg_patch0,1), size(neg_patch0,2)*size(neg_patch0,3) );
neg_patch0 = normalizeMat( neg_patch0 );

templates_CheckSample_num = size( templates_CheckSample,3 ); %用于检测样本的模板数目
patch_dic0 = GetPatch( templates_CheckSample, patch_size, patch_step );
patch_dic0 = reshape( patch_dic0, prod(patch_size), prod(patch_num)*templates_CheckSample_num );
patch_dic0 = normalizeMat(patch_dic0);%对字典标准化处理


param.L = size(neg_patch0,1);
neg_patch0_coef = mexLasso (neg_patch0, patch_dic0, param);%求所有patch块的稀疏系数
neg_patch0_coef = full(neg_patch0_coef);

ww = eye( prod(patch_num) );
ww = repmat(ww,templates_CheckSample_num,neg_num);
neg_rec_err = sum( ( neg_patch0 - patch_dic0*( ww.*neg_patch0_coef ) ).^2 );%重建误差
neg_rec_err = reshape( neg_rec_err, prod(patch_num), neg_num );
neg_rec_err = sum( neg_rec_err );

neg_remove_id = neg_rec_err<4.0;