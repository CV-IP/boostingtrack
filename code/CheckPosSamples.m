%% ���ɼ����������Ƿ���ʣ��Ѳ����ʵı�ǳ���


pos0_num = size(pos_sample_set,2); %����������������Ŀ
templates_CheckSample_num = size( templates_CheckSample,3 ); %���ڼ��������ģ����Ŀ

pos0 = reshape( pos_sample_set, template_size(1), template_size(2), pos0_num );
pos_patch0 = GetPatch( pos0, patch_size, patch_step );
pos_patch0 = reshape( pos_patch0, size(pos_patch0,1), size(pos_patch0,2)*size(pos_patch0,3) );
pos_patch0 = normalizeMat( pos_patch0 );

patch_dic0 = GetPatch( templates_CheckSample, patch_size, patch_step );
patch_dic0 = reshape( patch_dic0, prod(patch_size), prod(patch_num)*templates_CheckSample_num );
patch_dic0 = normalizeMat(patch_dic0);%���ֵ��׼������

param.L = size(pos_patch0,1);
pos_patch0_coef = mexLasso (pos_patch0, patch_dic0, param);%������patch���ϡ��ϵ��
pos_patch0_coef = full(pos_patch0_coef);

ww = eye( prod(patch_num) );
ww = repmat(ww,templates_CheckSample_num,pos0_num);
pos_rec_err = sum( ( pos_patch0 - patch_dic0*( ww.*pos_patch0_coef ) ).^2 );%�ؽ����
pos_rec_err = reshape( pos_rec_err, prod(patch_num), pos0_num );
pos_rec_err = sum( pos_rec_err );

pos_remove_id = pos_rec_err>3.5; %�ҳ������ʵ�����

%-----����templates_CheckSample---
if mod(f,15) == 0
    rand1 = randi( templates_CheckSample_num ); %���滻����ģ��
    rand2 = randi( pos0_num );
    while pos_rec_err(rand2) > 3.5
        rand2 = randi( pos0_num );
    end
    temp_result = pos_sample_set(:,rand2);
    temp_result = reshape( temp_result, 32, 32 );
    templates_CheckSample(:,:,rand1) = temp_result;
end







