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
InitVideo;  %��Ƶ���еĳ�ʼ������
%------Դ��Ƶλ��-------------
datapath = ['.\data\' title '\'];%Դ��Ƶ�����ļ���
if exist(datapath,'dir') == 0   %Դ��Ƶ�ļ��в����ڣ���ʾ
    disp('������Դ���ݣ�');
    break;
end
%------���ٽ������λ��-------
resultDir = [ 'Result_' int2str(xy_motion) '\' ]; %��ǰ���ٽ����ŵ��ļ���
resultpath = [ resultDir title '\' ];  
if exist(resultpath,'dir') == 0 %������浱ǰ���ٽ�����ļ��в����ڣ��򴴽�һ��
    mkdir(resultpath);
end

template_size = [32,32]; %ģ���С
template_num = 8;%ģ����

%---��ʼ�����������Ҳ����ѷ������
affparam = [p(1), p(2), p(3)/template_size(1), p(5), p(4)/p(3), 0];

%---�õ���Ƶ���е�֡��
frame_info = importdata( [datapath 'datainfo.txt'] );%��datainfo.txt�ж�ȡ֡����Ϣ�����ߡ�֡��
frame_num = frame_info(3);%֡����Ŀ
frame_width = frame_info(1);%֡��
frame_height = frame_info(2);%֡��

%---patch��Ĵ�С
patch_size = [16,16];
patch_step = 8;
patch_num(1) = length( patch_size(1)/2 : patch_step : (template_size(1)-patch_size(1)/2) );%ÿһ�а�����patch��
patch_num(2) = length( patch_size(2)/2 : patch_step : (template_size(2)-patch_size(1)/2) );%ÿһ�а�����patch��

%---����PCAҪ�õ��Ĳ���
tmpl.mean = [];
tmpl.basis = [];
tmpl.eigval = [];
tmpl.num = 0;

%---һ�������Ҫ�õ��Ĳ���
param.lambda = 0.01;
param.lambda2 = 0;
param.mode = 2;

%---���ؽ����ʱ�õ��Ĳ��������ʵ�����
beta = 0.01;
gamma = 5;

%-------��������ز���---------------
R = 60;%ѧϰ��������������
weak_classifier = cell( 1, R );

T = 45;%ѡ���������������
selected_weak_classifier = cell(1, T); 

%-----ÿ��ѵ����������ѡ���patch--------
SelectPatches;
features_patch = SeqSet;%

%---���������ĳ�ʼ��-------
pos_num = 9; %ÿ֡ѡ�������������Ŀ
neg_num = 150; %����ѵ���ĸ���������Ŀ
train_pos_num = 100;%����ѵ������������Ŀ

%-------��ʼ��ģ��Ȩ��-----------------
template_weight = ones( 1,template_num ); %ÿ��ģ���Ӧһ��Ȩ��


%% ��Ҷ˹��������������
%---���±�Ҷ˹������ʱ���õ��Ĳ���
lambda = 0.85;

%---���������ĳ�ʼ��
CT_pos_num = 200; %����������Ŀ
CT_neg_num = 200; %����������Ŀ

CT_M = prod(template_size);%��������ά���ά��

mean0 = zeros(CT_M,1);%�������ľ�ֵ
mean1 = zeros(CT_M,1);%�������ľ�ֵ
sig0 = ones(CT_M,1);  %�������ı�׼��
sig1 = ones(CT_M,1);  %�������ı�׼��

%% �����һ֡
img_color = imread( [datapath '1.jpg'] );
if size(img_color,3) == 3
    img = double(rgb2gray(img_color))/256;
else
    img = double(img_color)/256;
end

%---��ʼ����ֵ�ͷ���
[CT_pos_samples, CT_neg_samples] = CT_GetSamples(img, affparam, template_size, CT_pos_num, CT_neg_num);
%��ȡ�������͸�����
[mean1, mean0, sig1, sig0] = CT_UpdateClassifer(CT_pos_samples, CT_neg_samples, mean1, mean0, sig1, sig0, 0);
%��ʼ�����������ľ�ֵ�ͷ���

%---��ʼ����������
all_affparam = repmat( affparam(:),1,particle_num );%��ʼ���������ӵķ������
all_affparam = all_affparam + randn(6,particle_num).*repmat( affsig(:),1,particle_num );
all_affparam(:,end) = affparam(:);%ʹ���һ�����ӵķ���������ڵ�ǰ֡����ѷ������
CheckParticle;%��������Ƿ����

%---����PCAҪ�õ��ĸ���������
affimg = warpimg( img, affparam2mat(affparam), template_size );
templates = zeros(template_size(1),template_size(2),template_num);%��ʼ��ģ�弯
templates(:,:,1) = affimg;
update_vectors = affimg(:);  %��̬���

% ---��ʾ��һ֡
ShowResult(img_color,affparam,1,template_size);

%------�����һ֡���ٽ��----------------------------------
path = [resultpath int2str(1) '.jpg'];%д��ͼƬ��·����Ϣ
fff = getframe(gcf);%�õ�ͼ��ľ��
imwrite(fff.cdata,path);%д��ͼƬ
if exist([resultDir title '\PosInfo.txt'],'file') ~= 0   %λ����Ϣ�ļ��Ѵ��ڣ���ɾ��
    dos( ['del ' pwd  '\' resultDir title '\PosInfo.txt'] ); %ɾ��
    if exist([resultDir title '\PosInfo.txt'],'file') == 0
        disp('ɾ��PosInfo.txt�ɹ���');
    else
        disp('ɾ��PosInfo.txtʧ�ܣ�');
    end
else
    disp('PosInfo.txt���½���');
end

SavePosInfo(resultpath,affparam,template_size);
%% ��ʼ����
%-------��ʼ����������------------------
pos_sample_set = []; %��ʼ����������
temp = zeros(6,9);
temp(1,:) = [-1 1  0 0 -1 1 -1 1 0];
temp(2,:) = [ 0 0 -1 1 -1 1 1 -1 0];
pos_sample_affparam = repmat(affparam(:),1,pos_num);
pos_sample_affparam = pos_sample_affparam + temp;%�õ���������9���������
pos_samples = warpimg( img, affparam2mat( pos_sample_affparam ), template_size );
%�ɷ�������õ�9��������,32*32*9
pos_samples = reshape( pos_samples, prod(template_size), pos_num );
pos_sample_set = [pos_sample_set, pos_samples];%����µ�������

for f=2:frame_num
    img_color = imread( [datapath int2str(f) '.jpg'] );
    if size(img_color,3) == 3
        img = double(rgb2gray(img_color))/256;
    else
        img = double(img_color)/256;
    end
    
    Y = warpimg( img, affparam2mat(all_affparam), template_size );%����������Ӷ�Ӧ�ĺ�ѡĿ��
    YY = reshape(Y,prod(template_size),particle_num);%�Ժ�ѡĿ��������
    
    if (f<=template_num) %�ռ�10��ģ��͸�������
        likelihood = CT_GetLikelihood(YY,mean1,mean0,sig1,sig0);%������к�ѡĿ��Ĺ۲���Ȼֵ
        [likelihood,sort_id] = sort(likelihood,'descend');%����Ȼֵ����
        all_affparam = all_affparam(:,sort_id);%�����еķ������������Ӧ������
        Y = Y(:,:,sort_id);%�Ժ�ѡĿ�������Ӧ������
        YY = YY(:,sort_id);
        
        affparam = all_affparam(:,1);%�����ѵķ������
        templates(:,:,f) = Y(:,:,1);
        update_vectors = [update_vectors , YY(:,1) ];
        
        %---------��ȡ������-----------------------------
        pos_sample_affparam = repmat(affparam(:),1,pos_num);
        pos_sample_affparam = pos_sample_affparam + temp;%�õ���������9���������
        pos_samples = warpimg( img, affparam2mat( pos_sample_affparam ), template_size );
        %�ɷ�������õ�9��������,32*32*9
        pos_samples = reshape( pos_samples, prod(template_size), pos_num );
        pos_sample_set = [pos_sample_set, pos_samples];%����µ�������
        
        %---���±�Ҷ˹������-------------------------------
        [CT_pos_samples, CT_neg_samples] = CT_GetSamples(img, affparam, template_size, CT_pos_num, CT_neg_num);%��ȡ�������͸�����
        [mean1, mean0, sig1, sig0] = CT_UpdateClassifer(CT_pos_samples, CT_neg_samples, mean1, mean0, sig1, sig0, lambda);
    end
    
    %---���ռ���10��ģ��󣬽����ֵ��ʼ��������PCA��ʼ����ѵ����������
    if (f==template_num) 
        patch_dic = GetPatch(templates,patch_size,patch_step);%144*25*10
        patch_dic = reshape(patch_dic,prod(patch_size),prod(patch_num)*template_num);%144*250,�õ�patch�ֵ�
        patch_dic = normalizeMat(patch_dic);%���ֵ��׼������
        
        tmpl.mean = mean(update_vectors,2);%��ʼ������PCA��ֵ
        [tmpl.basis,tmpl.eigval,tmpl.mean,tmpl.num] = sklm(update_vectors,tmpl.basis,tmpl.eigval,tmpl.mean,tmpl.num);
        %��������PCA�㷨
        update_vectors = [];
        
        templates_CheckSample = templates; %���ڼ��������ģ�壬ֻ��CheckPosSamples��CheckNegSamples����ʹ�á�
        %-------------ѵ����������---------------
        [~, neg_samples] = GetSamples(img, affparam, template_size, pos_num, neg_num, affsig);%��ȡ������
        neg_samples = reshape( neg_samples, prod(template_size), neg_num );
        all_samples = [pos_sample_set,neg_samples];
        all_samples = reshape( all_samples, template_size(1), template_size(2), size(all_samples,2) );%����������
 
        all_patches = GetPatch( all_samples, patch_size, patch_step );%256*9*154
        [size1, size2, size3] = size(all_patches);
        all_patches = reshape( all_patches, size1, size2*size3 );
        all_patches = normalizeMat( all_patches );%��patch������׼������
        
        param.L = size(all_patches,1);
        patch_coef = mexLasso( all_patches, patch_dic, param);
        patch_coef = full( patch_coef );
        
        patch_coef = reshape( patch_coef, size(patch_coef,1), prod(patch_num), size3 );        

        X = patch_coef( :,:,1:size(pos_sample_set,2) );%��patch��ϵ��
        Y = patch_coef( :,:,size(pos_sample_set,2)+1:end );%��patch��ϵ��
        [weak_classifier, selected_weak_classifier] =MyAdaBoost(X, Y, R, T, features_patch);
    end
    
    %---�ӵ�11֡��ʼִ�б��ĵ��㷨
    if (f > template_num)
        Y_patch = GetPatch(Y,patch_size,patch_step);%�����еĺ�ѡĿ����з�patch
        Y_patch = reshape(Y_patch, prod(patch_size),prod(patch_num)*particle_num);%�õ�patch����144*1250
        Y_patch = normalizeMat(Y_patch);%��patch������׼������
        
        param.L = size(Y_patch,1);
        Y_patch_coef = mexLasso (Y_patch, patch_dic, param);%������patch���ϡ��ϵ��.250*1250
        Y_patch_coef = full(Y_patch_coef);
        
        %---��ÿ��patch�����Ȼֵ----
        w = eye( prod(patch_num) );
        w = repmat(w,template_num,particle_num);
        rec_err1 = sum( ( Y_patch - patch_dic*( w.*Y_patch_coef ) ).^2 );%�ؽ����
        rec_err2 = sum( patch_dic*((1-w).*Y_patch_coef) ); %�ͷ���
        rec_err = rec_err1 + beta*rec_err2;
        pro = exp(-5*rec_err);%ÿ��patch�����Ȼֵ
        
        %---���ؽ�������Ȼֵ------------
        pro = reshape(pro,prod(patch_num),particle_num);%��ͬһ����ѡĿ���patch������Ӧ����Ȼֵ����ͬһ��
        likelihood1 = sum(pro);
        likelihood1 = likelihood1/sum(likelihood1);
        
        %---��adaboost����������Ȼֵ----
        Y_patch_coef = reshape( Y_patch_coef, size( Y_patch_coef,1 ), prod(patch_num), particle_num );

        rate = zeros( 1, particle_num );
        for tmp_i = 1:particle_num
            rate(tmp_i) = MyAdaboostClassify( features_patch, selected_weak_classifier, Y_patch_coef(:, :, tmp_i), 1 );
        end 
        rate = exp( 1*rate );
        likelihood2 = rate/sum(rate);
        %-------������Ȼֵ--------
        likelihood = likelihood1 .*likelihood2;
        
        %---�����ѷ������
        [~,max_id] = max(likelihood);
        affparam = all_affparam(:,max_id);%�����ѵķ������
        
        %---------����Ѻ�ѡĿ����Χѡȡ9������������ӵ�����������--------
        pos_sample_affparam = repmat(affparam(:),1,pos_num);
        pos_sample_affparam = pos_sample_affparam + temp;%�õ���������9���������
        pos_samples = warpimg( img, affparam2mat( pos_sample_affparam ), template_size );
        %�ɷ�������õ�9��������,32*32*9
        pos_samples = reshape( pos_samples, prod(template_size), pos_num );
        pos_sample_set = [pos_sample_set, pos_samples];%����µ�������
        CheckPosSamples; % ����������Ƿ����
        pos_sample_set( :,pos_remove_id ) = []; %ȥ�������ʵ�������
        
        possize = size( pos_sample_set,2 );
        if (possize > train_pos_num ) 
            % ������������е�����������train_pos_num����ȥ����ǰ��ģ�possize-train_pos_num��������
            remove_num = possize - train_pos_num;
            pos_sample_set(:,1:remove_num) = [];
        end
        
        %---���������,�������¡�����PCA���Ļ�����
        update_vector = YY(:,max_id);
        [~, ~, E] = LSS( update_vector-tmpl.mean, tmpl.basis,tmpl.basis');
        update_vector = (E==0).*update_vector + (E~=0).*tmpl.mean;
        update_vectors = [update_vectors, update_vector];
        
        template_set_vec = reshape( templates, prod(template_size), template_num );%ȫ��ģ��������
        template_set_vec = normalizeMat( template_set_vec );
        update_vector = normalizeMat( update_vector );
        angle = zeros(1,template_num);
        for tt = 1:template_num
            angle(tt) = interangle( update_vector,template_set_vec(:,tt) );         
        end
        template_weight = template_weight.*exp(-angle);  %����ģ��Ȩ��
        template_weight = template_weight/sum(template_weight);
        
        %---ִ������PCA��ͬʱ����patch�ֵ�ĸ���
        if size(update_vectors,2) == interval
            
            [~, coef, E] = LSS( YY(:,max_id)-tmpl.mean, tmpl.basis,tmpl.basis');
            rec_template = tmpl.basis*coef + tmpl.mean;%�ؽ�����ģ��
            rec_template = reshape(rec_template,template_size);%ת����32*32��ģ�壬�����ȡpatch
            
            [tmpl.basis,tmpl.eigval,tmpl.mean,tmpl.num] = sklm(update_vectors,tmpl.basis,tmpl.eigval,tmpl.mean,tmpl.num);
            %��������PCA�㷨
            update_vectors = [];
            if size(tmpl.basis,2)>template_num
                tmpl.basis = tmpl.basis(:, 1:template_num);
                tmpl.eigval = tmpl.eigval(1:template_num);
            end
            
            %-----���Ҫ�滻��ģ��-----
            [~,min_id] = min( template_weight );
            templates(:,:,min_id) = rec_template;%��ģ���滻��ģ��
        
            template_weight(min_id) = 0;
            template_weight(min_id) = median(template_weight);
            template_weight = template_weight/sum(template_weight);
            while ( max(template_weight) > 0.3 )
                big = find(template_weight>0.3);
                template_weight(big) = 0.3;
                template_weight = template_weight/sum(template_weight);
            end
        
            %-------����ѵ���ֵ�---------- 
            template_patch = GetPatch( templates, patch_size, patch_step );%256*9*9
            template_patch = reshape( template_patch, prod(patch_size), prod(patch_num)*template_num );
            patch_dic = normalizeMat(template_patch);%���ֵ��׼������
            
            %-------------����ѵ����������---------------
            [~, neg_samples] = GetSamples(img, affparam, template_size, pos_num, neg_num, affsig);%��ȡ������
            CheckNegSamples; % ��鸺�����Ƿ����
            neg_samples = reshape( neg_samples, prod(template_size), neg_num );
            neg_samples( :,neg_remove_id ) = [];%ȥ�������ʵĸ�����
            all_samples = [pos_sample_set,neg_samples];
            all_samples = reshape( all_samples, template_size(1), template_size(2), size(all_samples,2) );
            %����������

            all_patches = GetPatch( all_samples, patch_size, patch_step );%256*9*154
            [size1, size2, size3] = size(all_patches);
            all_patches = reshape( all_patches, size1, size2*size3 );
            all_patches = normalizeMat( all_patches );%��patch������׼������
        
            param.L = size(all_patches,1);
            patch_coef = mexLasso( all_patches, patch_dic, param);
            patch_coef = full( patch_coef );
            patch_coef = reshape( patch_coef, size(patch_coef,1), prod(patch_num), size3 );
            
            X = patch_coef( :,:,1:size(pos_sample_set,2) );%��patch��ϵ��
            Y = patch_coef( :,:,size(pos_sample_set,2)+1:end );%��patch��ϵ��
            [weak_classifier, selected_weak_classifier] =MyAdaBoost(X, Y, R, T, features_patch);
        end 
    end
    
    
    %-------�ز���------------------------------------------
    all_affparam = repmat( affparam(:),1,particle_num );%��ʼ���������ӵķ������
    all_affparam = all_affparam + randn(6,particle_num).*repmat( affsig(:),1,particle_num );
    all_affparam(:,end) = affparam(:);%ʹ���һ�����ӵķ���������ڵ�ǰ֡����ѷ������
    CheckParticle;%��������Ƿ����
    
    %------��ʾ���ٽ��---------------------------------- 
    ShowResult(img_color,affparam,f,template_size);
%     
%     ------������ٽ��----------------------------------
    path = [resultpath int2str(f) '.jpg'];%д��ͼƬ��·����Ϣ
    fff = getframe(gcf);%�õ�ͼ��ľ��
    imwrite(fff.cdata,path);%д��ͼƬ
    
    SavePosInfo(resultpath,affparam,template_size);
end














