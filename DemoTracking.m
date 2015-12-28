% B. Ma, J. Shen, Y. Liu, H. Hu, L. Shao, X. Li, Visual tracking using 
% strong classifier and structural local sparse descriptors, 
% IEEE Trans. on Multimedia, ??, 2015

clear all;
close all;
addpath('code');
addpath('data');
addpath('mex');
addpath('mexLasso');

xy_motion = 8; %the range of movement in directions of x and y
%------parameters ---------------
affsig = [ xy_motion, xy_motion, 0.005, 0.00, 0.00, 0.00 ]; % motion parameters
particle_num = 600; % particles
interval = 5;       % update interval

%------tracking video clips----------
title = 'doll';
%% DemoTracking;
%-----------------------------

%% Initialization
InitVideo;  %视频序列的初始化设置
%------源视频位置-------------
datapath = ['.\data\' title '\'];%源视频所在文件夹
if exist(datapath,'dir') == 0   %源视频文件夹不存在，提示
    disp('不存在源数据！');
    break;
end
%------跟踪结果保存位置-------
resultDir = [ 'Result_' int2str(xy_motion) '\' ]; %当前跟踪结果存放的文件夹
resultpath = [ resultDir title '\' ];  
if exist(resultpath,'dir') == 0 %如果保存当前跟踪结果的文件夹不存在，则创建一个
    mkdir(resultpath);
end

template_size = [32,32]; %模板大小
template_num = 8;%模板数

%---初始化仿射参数，也是最佳仿射参数
affparam = [p(1), p(2), p(3)/template_size(1), p(5), p(4)/p(3), 0];

%---得到视频序列的帧数
frame_info = importdata( [datapath 'datainfo.txt'] );%从datainfo.txt中读取帧的信息，宽、高、帧数
frame_num = frame_info(3);%帧的数目
frame_width = frame_info(1);%帧宽
frame_height = frame_info(2);%帧高

%---patch块的大小
patch_size = [16,16];
patch_step = 8;
patch_num(1) = length( patch_size(1)/2 : patch_step : (template_size(1)-patch_size(1)/2) );%每一行包括的patch数
patch_num(2) = length( patch_size(2)/2 : patch_step : (template_size(2)-patch_size(1)/2) );%每一列包括的patch数

%---增量PCA要用到的参数
tmpl.mean = [];
tmpl.basis = [];
tmpl.eigval = [];
tmpl.num = 0;

%---一范数求解要用到的参数
param.lambda = 0.01;
param.lambda2 = 0;
param.mode = 2;

%---求重建误差时用到的参数，可适当调节
beta = 0.01;
gamma = 5;

%-------分类器相关参数---------------
R = 60;%学习的弱分类器个数
weak_classifier = cell( 1, R );

T = 45;%选择的弱分类器个数
selected_weak_classifier = cell(1, T); 

%-----每次训练分类器所选择的patch--------
SelectPatches;
features_patch = SeqSet;%

%---正负样本的初始化-------
pos_num = 9; %每帧选择的正样本的数目
neg_num = 150; %参与训练的负样本的数目
train_pos_num = 100;%参与训练的正样本数目

%-------初始化模板权重-----------------
template_weight = ones( 1,template_num ); %每个模板对应一个权重


%% 贝叶斯分类器参数设置
%---更新贝叶斯分类器时，用到的参数
lambda = 0.85;

%---正负样本的初始化
CT_pos_num = 200; %正样本的数目
CT_neg_num = 200; %负样本的数目

CT_M = prod(template_size);%对样本降维后的维数

mean0 = zeros(CT_M,1);%负样本的均值
mean1 = zeros(CT_M,1);%正样本的均值
sig0 = ones(CT_M,1);  %负样本的标准差
sig1 = ones(CT_M,1);  %正样本的标准差

%% 处理第一帧
img_color = imread( [datapath '1.jpg'] );
if size(img_color,3) == 3
    img = double(rgb2gray(img_color))/256;
else
    img = double(img_color)/256;
end

%---初始化均值和方差
[CT_pos_samples, CT_neg_samples] = CT_GetSamples(img, affparam, template_size, CT_pos_num, CT_neg_num);
%获取正样本和负样本
[mean1, mean0, sig1, sig0] = CT_UpdateClassifer(CT_pos_samples, CT_neg_samples, mean1, mean0, sig1, sig0, 0);
%初始化正负样本的均值和方差

%---初始化所有样本
all_affparam = repmat( affparam(:),1,particle_num );%初始化所有粒子的仿射参数
all_affparam = all_affparam + randn(6,particle_num).*repmat( affsig(:),1,particle_num );
all_affparam(:,end) = affparam(:);%使最后一个粒子的仿射参数等于当前帧的最佳仿射参数
CheckParticle;%检查粒子是否合适

%---增量PCA要用到的更新向量集
affimg = warpimg( img, affparam2mat(affparam), template_size );
templates = zeros(template_size(1),template_size(2),template_num);%初始化模板集
templates(:,:,1) = affimg;
update_vectors = affimg(:);  %动态添加

% ---显示第一帧
ShowResult(img_color,affparam,1,template_size);

%------保存第一帧跟踪结果----------------------------------
path = [resultpath int2str(1) '.jpg'];%写入图片的路径信息
fff = getframe(gcf);%得到图像的句柄
imwrite(fff.cdata,path);%写入图片
if exist([resultDir title '\PosInfo.txt'],'file') ~= 0   %位置信息文件已存在，则删除
    dos( ['del ' pwd  '\' resultDir title '\PosInfo.txt'] ); %删除
    if exist([resultDir title '\PosInfo.txt'],'file') == 0
        disp('删除PosInfo.txt成功！');
    else
        disp('删除PosInfo.txt失败！');
    end
else
    disp('PosInfo.txt被新建！');
end

SavePosInfo(resultpath,affparam,template_size);
%% 开始跟踪
%-------初始化正样本集------------------
pos_sample_set = []; %初始化正样本集
temp = zeros(6,9);
temp(1,:) = [-1 1  0 0 -1 1 -1 1 0];
temp(2,:) = [ 0 0 -1 1 -1 1 1 -1 0];
pos_sample_affparam = repmat(affparam(:),1,pos_num);
pos_sample_affparam = pos_sample_affparam + temp;%得到正样本的9个仿射参数
pos_samples = warpimg( img, affparam2mat( pos_sample_affparam ), template_size );
%由仿射参数得到9个正样本,32*32*9
pos_samples = reshape( pos_samples, prod(template_size), pos_num );
pos_sample_set = [pos_sample_set, pos_samples];%添加新的正样本

for f=2:frame_num
    img_color = imread( [datapath int2str(f) '.jpg'] );
    if size(img_color,3) == 3
        img = double(rgb2gray(img_color))/256;
    else
        img = double(img_color)/256;
    end
    
    Y = warpimg( img, affparam2mat(all_affparam), template_size );%求得所有粒子对应的候选目标
    YY = reshape(Y,prod(template_size),particle_num);%对候选目标向量化
    
    if (f<=template_num) %收集10个模板和更新向量
        likelihood = CT_GetLikelihood(YY,mean1,mean0,sig1,sig0);%求得所有候选目标的观测似然值
        [likelihood,sort_id] = sort(likelihood,'descend');%对似然值排序
        all_affparam = all_affparam(:,sort_id);%对所有的仿射参数进行相应的排序
        Y = Y(:,:,sort_id);%对候选目标进行相应的排序
        YY = YY(:,sort_id);
        
        affparam = all_affparam(:,1);%求出最佳的仿射参数
        templates(:,:,f) = Y(:,:,1);
        update_vectors = [update_vectors , YY(:,1) ];
        
        %---------获取正样本-----------------------------
        pos_sample_affparam = repmat(affparam(:),1,pos_num);
        pos_sample_affparam = pos_sample_affparam + temp;%得到正样本的9个仿射参数
        pos_samples = warpimg( img, affparam2mat( pos_sample_affparam ), template_size );
        %由仿射参数得到9个正样本,32*32*9
        pos_samples = reshape( pos_samples, prod(template_size), pos_num );
        pos_sample_set = [pos_sample_set, pos_samples];%添加新的正样本
        
        %---更新贝叶斯分类器-------------------------------
        [CT_pos_samples, CT_neg_samples] = CT_GetSamples(img, affparam, template_size, CT_pos_num, CT_neg_num);%获取正样本和负样本
        [mean1, mean0, sig1, sig0] = CT_UpdateClassifer(CT_pos_samples, CT_neg_samples, mean1, mean0, sig1, sig0, lambda);
    end
    
    %---当收集到10个模板后，进行字典初始化和增量PCA初始化，训练弱分类器
    if (f==template_num) 
        patch_dic = GetPatch(templates,patch_size,patch_step);%144*25*10
        patch_dic = reshape(patch_dic,prod(patch_size),prod(patch_num)*template_num);%144*250,得到patch字典
        patch_dic = normalizeMat(patch_dic);%对字典标准化处理
        
        tmpl.mean = mean(update_vectors,2);%初始化增量PCA均值
        [tmpl.basis,tmpl.eigval,tmpl.mean,tmpl.num] = sklm(update_vectors,tmpl.basis,tmpl.eigval,tmpl.mean,tmpl.num);
        %调用增量PCA算法
        update_vectors = [];
        
        templates_CheckSample = templates; %用于检测样本的模板，只在CheckPosSamples和CheckNegSamples里面使用。
        %-------------训练弱分类器---------------
        [~, neg_samples] = GetSamples(img, affparam, template_size, pos_num, neg_num, affsig);%获取负样本
        neg_samples = reshape( neg_samples, prod(template_size), neg_num );
        all_samples = [pos_sample_set,neg_samples];
        all_samples = reshape( all_samples, template_size(1), template_size(2), size(all_samples,2) );%正负样本集
 
        all_patches = GetPatch( all_samples, patch_size, patch_step );%256*9*154
        [size1, size2, size3] = size(all_patches);
        all_patches = reshape( all_patches, size1, size2*size3 );
        all_patches = normalizeMat( all_patches );%对patch集，标准化处理
        
        param.L = size(all_patches,1);
        patch_coef = mexLasso( all_patches, patch_dic, param);
        patch_coef = full( patch_coef );
        
        patch_coef = reshape( patch_coef, size(patch_coef,1), prod(patch_num), size3 );        

        X = patch_coef( :,:,1:size(pos_sample_set,2) );%正patch块系数
        Y = patch_coef( :,:,size(pos_sample_set,2)+1:end );%负patch块系数
        [weak_classifier, selected_weak_classifier] =MyAdaBoost(X, Y, R, T, features_patch);
    end
    
    %---从第11帧开始执行本文的算法
    if (f > template_num)
        Y_patch = GetPatch(Y,patch_size,patch_step);%对所有的候选目标进行分patch
        Y_patch = reshape(Y_patch, prod(patch_size),prod(patch_num)*particle_num);%得到patch集，144*1250
        Y_patch = normalizeMat(Y_patch);%对patch集，标准化处理
        
        param.L = size(Y_patch,1);
        Y_patch_coef = mexLasso (Y_patch, patch_dic, param);%求所有patch块的稀疏系数.250*1250
        Y_patch_coef = full(Y_patch_coef);
        
        %---求每个patch块的似然值----
        w = eye( prod(patch_num) );
        w = repmat(w,template_num,particle_num);
        rec_err1 = sum( ( Y_patch - patch_dic*( w.*Y_patch_coef ) ).^2 );%重建误差
        rec_err2 = sum( patch_dic*((1-w).*Y_patch_coef) ); %惩罚项
        rec_err = rec_err1 + beta*rec_err2;
        pro = exp(-5*rec_err);%每个patch块的似然值
        
        %---由重建误差得似然值------------
        pro = reshape(pro,prod(patch_num),particle_num);%把同一个候选目标的patch块所对应的似然值放在同一列
        likelihood1 = sum(pro);
        likelihood1 = likelihood1/sum(likelihood1);
        
        %---由adaboost分类器得似然值----
        Y_patch_coef = reshape( Y_patch_coef, size( Y_patch_coef,1 ), prod(patch_num), particle_num );

        rate = zeros( 1, particle_num );
        for tmp_i = 1:particle_num
            rate(tmp_i) = MyAdaboostClassify( features_patch, selected_weak_classifier, Y_patch_coef(:, :, tmp_i), 1 );
        end 
        rate = exp( 1*rate );
        likelihood2 = rate/sum(rate);
        %-------总体似然值--------
        likelihood = likelihood1 .*likelihood2;
        
        %---求出最佳仿射参数
        [~,max_id] = max(likelihood);
        affparam = all_affparam(:,max_id);%求出最佳的仿射参数
        
        %---------在最佳候选目标周围选取9个正样本，添加到正样本集中--------
        pos_sample_affparam = repmat(affparam(:),1,pos_num);
        pos_sample_affparam = pos_sample_affparam + temp;%得到正样本的9个仿射参数
        pos_samples = warpimg( img, affparam2mat( pos_sample_affparam ), template_size );
        %由仿射参数得到9个正样本,32*32*9
        pos_samples = reshape( pos_samples, prod(template_size), pos_num );
        pos_sample_set = [pos_sample_set, pos_samples];%添加新的正样本
        CheckPosSamples; % 检查正样本是否合适
        pos_sample_set( :,pos_remove_id ) = []; %去除不合适的正样本
        
        possize = size( pos_sample_set,2 );
        if (possize > train_pos_num ) 
            % 如果正样本集中的样本数多于train_pos_num，则去除最前面的（possize-train_pos_num）个样本
            remove_num = possize - train_pos_num;
            pos_sample_set(:,1:remove_num) = [];
        end
        
        %---求更新向量,用来更新“增量PCA”的基向量
        update_vector = YY(:,max_id);
        [~, ~, E] = LSS( update_vector-tmpl.mean, tmpl.basis,tmpl.basis');
        update_vector = (E==0).*update_vector + (E~=0).*tmpl.mean;
        update_vectors = [update_vectors, update_vector];
        
        template_set_vec = reshape( templates, prod(template_size), template_num );%全局模板向量化
        template_set_vec = normalizeMat( template_set_vec );
        update_vector = normalizeMat( update_vector );
        angle = zeros(1,template_num);
        for tt = 1:template_num
            angle(tt) = interangle( update_vector,template_set_vec(:,tt) );         
        end
        template_weight = template_weight.*exp(-angle);  %更新模板权重
        template_weight = template_weight/sum(template_weight);
        
        %---执行增量PCA，同时进行patch字典的更新
        if size(update_vectors,2) == interval
            
            [~, coef, E] = LSS( YY(:,max_id)-tmpl.mean, tmpl.basis,tmpl.basis');
            rec_template = tmpl.basis*coef + tmpl.mean;%重建的新模板
            rec_template = reshape(rec_template,template_size);%转换成32*32的模板，方便获取patch
            
            [tmpl.basis,tmpl.eigval,tmpl.mean,tmpl.num] = sklm(update_vectors,tmpl.basis,tmpl.eigval,tmpl.mean,tmpl.num);
            %调用增量PCA算法
            update_vectors = [];
            if size(tmpl.basis,2)>template_num
                tmpl.basis = tmpl.basis(:, 1:template_num);
                tmpl.eigval = tmpl.eigval(1:template_num);
            end
            
            %-----求出要替换的模板-----
            [~,min_id] = min( template_weight );
            templates(:,:,min_id) = rec_template;%新模板替换旧模板
        
            template_weight(min_id) = 0;
            template_weight(min_id) = median(template_weight);
            template_weight = template_weight/sum(template_weight);
            while ( max(template_weight) > 0.3 )
                big = find(template_weight>0.3);
                template_weight(big) = 0.3;
                template_weight = template_weight/sum(template_weight);
            end
        
            %-------重新训练字典---------- 
            template_patch = GetPatch( templates, patch_size, patch_step );%256*9*9
            template_patch = reshape( template_patch, prod(patch_size), prod(patch_num)*template_num );
            patch_dic = normalizeMat(template_patch);%对字典标准化处理
            
            %-------------重新训练弱分类器---------------
            [~, neg_samples] = GetSamples(img, affparam, template_size, pos_num, neg_num, affsig);%获取负样本
            CheckNegSamples; % 检查负样本是否合适
            neg_samples = reshape( neg_samples, prod(template_size), neg_num );
            neg_samples( :,neg_remove_id ) = [];%去掉不合适的负样本
            all_samples = [pos_sample_set,neg_samples];
            all_samples = reshape( all_samples, template_size(1), template_size(2), size(all_samples,2) );
            %正负样本集

            all_patches = GetPatch( all_samples, patch_size, patch_step );%256*9*154
            [size1, size2, size3] = size(all_patches);
            all_patches = reshape( all_patches, size1, size2*size3 );
            all_patches = normalizeMat( all_patches );%对patch集，标准化处理
        
            param.L = size(all_patches,1);
            patch_coef = mexLasso( all_patches, patch_dic, param);
            patch_coef = full( patch_coef );
            patch_coef = reshape( patch_coef, size(patch_coef,1), prod(patch_num), size3 );
            
            X = patch_coef( :,:,1:size(pos_sample_set,2) );%正patch块系数
            Y = patch_coef( :,:,size(pos_sample_set,2)+1:end );%负patch块系数
            [weak_classifier, selected_weak_classifier] =MyAdaBoost(X, Y, R, T, features_patch);
        end 
    end
    
    
    %-------重采样------------------------------------------
    all_affparam = repmat( affparam(:),1,particle_num );%初始化所有粒子的仿射参数
    all_affparam = all_affparam + randn(6,particle_num).*repmat( affsig(:),1,particle_num );
    all_affparam(:,end) = affparam(:);%使最后一个粒子的仿射参数等于当前帧的最佳仿射参数
    CheckParticle;%检查粒子是否合适
    
    %------显示跟踪结果---------------------------------- 
    ShowResult(img_color,affparam,f,template_size);
%     
%     ------保存跟踪结果----------------------------------
    path = [resultpath int2str(f) '.jpg'];%写入图片的路径信息
    fff = getframe(gcf);%得到图像的句柄
    imwrite(fff.cdata,path);%写入图片
    
    SavePosInfo(resultpath,affparam,template_size);
end














