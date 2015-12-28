function [A_pos, A_neg , pos_affpara] = CT_GetSamples( image, affpara, template_size, pos_num, neg_num )
%
%
%
%% �õ�pos_num��������
all_affpara = repmat( affpara(:), 1, pos_num);%��ʼ�������������ķ������

sigma = [1.5, 1.5, 0.0, 0.0, 0.0, 0.0];
all_affpara = all_affpara + randn(6,pos_num).*repmat( sigma(:), 1, pos_num);%�����еķ����������Ŷ�

pos_affpara = all_affpara;

all_affpara_mat = affparam2mat( all_affpara );%�����еķ����������ת�����õ�ת����ķ������
all_affimg = warpimg( image, all_affpara_mat, template_size );%��������������õ������ķ���ͼ

A_pos = reshape(all_affimg,prod(template_size),pos_num);

%% �õ�neg_num��������
candi_neg_num = neg_num*10;%��ѡ������������
all_affpara = repmat( affpara(:), 1, candi_neg_num);%��ʼ�����и������ķ������

affpara_mat = affparam2mat( affpara );
sigma = [ round( template_size(1)*affpara_mat(3) ) , round( template_size(1)*affpara_mat(3)*affpara(5) ) , 0.0, 0.0, 0.0, 0.0];
all_affpara = all_affpara + randn(6,candi_neg_num).*repmat( sigma(:), 1, candi_neg_num);%�����еķ����������Ŷ�

% ������и�������Ŀ�����ĵľ����Ƿ����һ����ֵ
%--���x����
dist_x = round( sigma(1)/4 );%������λ�õĺ�������룬��������������λ�õĺ��������Ҫ���ڸ�ֵ��
center_x = affpara(1);%����λ�õĺ�����
left = center_x - dist_x;%����߽�
right = center_x + dist_x;%���ұ߽�

dist_y = round( sigma(2)/4 );
center_y = affpara(2);
top = center_y - dist_y;%���ϱ߽�
bottom = center_y + dist_y;%���±߽�

id = all_affpara(1,:)<= right & all_affpara(1,:)>=left & all_affpara( 2,: )>= top & all_affpara( 2,: ) <= bottom; % ȥ��������ĸ�������
all_affpara(:,id) = [];
%----------------
dist_x = round( sigma(1) );%������λ�õĺ��������
center_x = affpara(1);%����λ�õĺ�����
left = center_x - dist_x;%����߽�
right = center_x + dist_x;%���ұ߽�

dist_y = round( sigma(2) );
center_y = affpara(2);
top = center_y - dist_y;%���ϱ߽�
bottom = center_y + dist_y;%���±߽�

id = all_affpara(1,:)<=left | all_affpara(1,:) >= right | all_affpara(2,:) <= top | all_affpara(2,:) >= bottom;% ȥ��������ĸ�������
all_affpara(:,id) = [];
%----------------
[img_h,img_w] = size(image);
id = ( all_affpara(1,:)<0 | all_affpara(1,:)>img_w | all_affpara(2,:)<0 | all_affpara(2,:)>img_h );% ȥ��������ĸ�������
all_affpara(:,id) = [];
num = size(all_affpara,2);
neg_id = unidrnd(num,[1,neg_num]);
all_affpara = all_affpara(:,neg_id);

all_affpara_mat = affparam2mat( all_affpara );%�����еķ����������ת�����õ�ת����ķ������
all_affimg = warpimg( image, all_affpara_mat, template_size );%��������������õ������ķ���ͼ

A_neg = reshape(all_affimg,prod(template_size),neg_num);

end

