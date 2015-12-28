%% 检查采集的正样本是否合适，把不合适的标记出来


pos0_num = size(pos_sample_set,2); %待检测的正样本的数目
templates_CheckSample_num = size( templates_CheckSample,3 ); %用于检测样本的模板数目

pos0 = reshape( pos_sample_set, template_size(1), template_size(2), pos0_num );
pos_patch0 = GetPatch( pos0, patch_size, patch_step );
pos_patch0 = reshape( pos_patch0, size(pos_patch0,1), size(pos_patch0,2)*size(pos_patch0,3) );
pos_patch0 = normalizeMat( pos_patch0 );

patch_dic0 = GetPatch( templates_CheckSample, patch_size, patch_step );
patch_dic0 = reshape( patch_dic0, prod(patch_size), prod(patch_num)*templates_CheckSample_num );
patch_dic0 = normalizeMat(patch_dic0);%对字典标准化处理

param.L = size(pos_patch0,1);
pos_patch0_coef = mexLasso (pos_patch0, patch_dic0, param);%求所有patch块的稀疏系数
pos_patch0_coef = full(pos_patch0_coef);

ww = eye( prod(patch_num) );
ww = repmat(ww,templates_CheckSample_num,pos0_num);
pos_rec_err = sum( ( pos_patch0 - patch_dic0*( ww.*pos_patch0_coef ) ).^2 );%重建误差
pos_rec_err = reshape( pos_rec_err, prod(patch_num), pos0_num );
pos_rec_err = sum( pos_rec_err );

pos_remove_id = pos_rec_err>3.5; %找出不合适的样本

%-----更新templates_CheckSample---
if mod(f,15) == 0
    rand1 = randi( templates_CheckSample_num ); %被替换掉的模板
    rand2 = randi( pos0_num );
    while pos_rec_err(rand2) > 3.5
        rand2 = randi( pos0_num );
    end
    temp_result = pos_sample_set(:,rand2);
    temp_result = reshape( temp_result, 32, 32 );
    templates_CheckSample(:,:,rand1) = temp_result;
end







