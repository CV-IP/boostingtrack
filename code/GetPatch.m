function patch = GetPatch( all_affimg, patch_size, patch_step )
%GetPatch：把图像按指定的patch块大小，分割成一系列patch块，并向量化patch块
%all_affimg:要分割patch的图像，w*h*n
%patch_size：patch块大小
%patch_step:
%patch：向量化的patch块，d*m*n
%% 
sample_num = size( all_affimg, 3 );
template_size = size( all_affimg(:,:,1) );
patch_num(1) = length( patch_size(1)/2 : patch_step : (template_size(1)-patch_size(1)/2) );%每一行包括的patch数
patch_num(2) = length( patch_size(2)/2 : patch_step : (template_size(2)-patch_size(1)/2) );%每一列包括的patch数
patch = zeros( prod(patch_size), prod(patch_num), sample_num );

x = patch_size(1)/2;
y = patch_size(2)/2;
patch_centerx = x : patch_step : ( template_size(1)-x ) ;% 所有patch的中心x坐标
patch_centery = y : patch_step : ( template_size(2)-y ) ;% 所有patch的中心y坐标

l = 1;
for j = 1:patch_num(1)
    for k = 1:patch_num(2)
        data = all_affimg( patch_centerx(j)-x+1:patch_centerx(j)+x , patch_centery(k)-y+1:patch_centery(k)+y , : );
        data = reshape( data, prod(patch_size), 1, sample_num );
        patch(:,l,:) = data;
        l = l + 1;
    end
end

end

