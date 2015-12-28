function [A_pos, A_neg] = GetSamples( image, affpara, template_size, pos_num, neg_num, affsig )
%
%
%
%% 得到pos_num个正样本
all_affpara = repmat( affpara(:), 1, pos_num);%初始化所有正样本的仿射参数

sigma = [0.75, 0.75, 0.0, 0.0, 0.0, 0.0];
all_affpara = all_affpara + randn(6,pos_num).*repmat( sigma(:), 1, pos_num);%对所有的仿射参数随机扰动

all_affpara_mat = affparam2mat( all_affpara );%对所有的仿射参数进行转换，得到转换后的仿射参数
A_pos = warpimg( image, all_affpara_mat, template_size );%由样本仿射参数得到样本的仿射图

% TestNeg( image, affpara, all_affpara, template_size );%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 得到neg_num个负样本
candi_neg_num = neg_num*10;%候选负样本样本数
all_affpara = repmat( affpara(:), 1, candi_neg_num);%初始化所有负样本的仿射参数
affsig(:,3:6) = 0;
all_affpara = all_affpara + 2*randn(6,candi_neg_num).*repmat( affsig(:), 1, candi_neg_num);%对所有的仿射参数随机扰动

%-----------检查所有负样本到目标中心的距离是否大于一个定值-----------------
dist_x = 6;%到中心位置的横坐标距离，（负样本到中心位置的横坐标距离要大于该值）
center_x = affpara(1);%中心位置的横坐标
left = center_x - dist_x;%内左边界
right = center_x + dist_x;%内右边界

dist_y = 6;
center_y = affpara(2);
top = center_y - dist_y;%内上边界
bottom = center_y + dist_y;%内下边界

id = all_affpara(1,:)<= right & all_affpara(1,:)>=left & all_affpara( 2,: )>= top & all_affpara( 2,: ) <= bottom; % 去除不合理的负样本点
all_affpara(:,id) = [];

%-----------检查负样本是否越界--------------------------
[img_h,img_w] = size(image);
box_w = round( affpara(3)*template_size(1) );
box_h = round( affpara(3)*template_size(1)*affpara(5) );

id = ( all_affpara(1,:)-box_w/2-1<0 | all_affpara(1,:)+box_w/2+1>img_w | all_affpara(2,:)-box_h/2-1<0 | all_affpara(2,:)+box_h/2+1>img_h );% 去除不合理的负样本点
all_affpara(:,id) = [];
num = size(all_affpara,2);
neg_id = unidrnd(num,[1,neg_num]);
all_affpara = all_affpara(:,neg_id);

all_affpara_mat = affparam2mat( all_affpara );%对所有的仿射参数进行转换，得到转换后的仿射参数
A_neg = warpimg( image, all_affpara_mat, template_size );%由样本仿射参数得到样本的仿射图
% TestNeg( image, affpara, all_affpara, template_size );%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

