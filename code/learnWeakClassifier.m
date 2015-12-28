function ret  = learnWeakClassifier( samples_weight , sub_samples_indx, feature_indx, features_patch, all_samples, all_samples_flag)
%%
%
%
%
%%
sub_samples_flag =  all_samples_flag( sub_samples_indx );%����ѵ����������ǩ
sub_samples = all_samples( :, :, sub_samples_indx ); %����ѵ��������          
X_indx = find( sub_samples_flag > 0 ); %��������Ӧ������
Y_indx = find( sub_samples_flag < 0 ); %��������Ӧ������

patch_index = features_patch( :,feature_indx );

sub_samples = sub_samples( :, patch_index, : );
% sub_samples = sum( sub_samples, 2 )./size(patch_index,1); % average pooling
% sub_samples = max( sub_samples,[],2 ); % max poolig
% sub_samples = reshape( sub_samples, size( sub_samples, 1), size( sub_samples, 3) );
sub_samples  = reshape(sub_samples,size( sub_samples, 1)*size( sub_samples, 2), size( sub_samples, 3) );%concatenate pooling


X = sub_samples(:, X_indx);%������
Y = sub_samples(:, Y_indx);%������
% 
% % [w_opt, w0] =  fisherClassifier(X,Y);%ѵ��fisher������
% 
%%
%-----------SVM-----------------------------------------------------------
% L1 =  ones( 1,size(X,2) );
% L2 = -ones( 1,size(Y,2) );
% L = [L1 L2];
% XY = [X,Y];
% Classifer = train( L', sparse(XY), '-s 0 -B 1 -c 10', 'col' );  %1/lamda = 10,lamda = 0.1
% global_w = Classifer.w;
% global_w = global_w'; 
% w_opt = global_w( 1: size(global_w)-1 );
% w0 = global_w( end );

%----------���ر�Ҷ˹--------------------------------------------
L1 =  ones( 1,size(X,2) );
L2 = -ones( 1,size(Y,2) );
XY = [X,Y]';%����
L = [L1,L2]';%��ǩ

%------���������й�һ��-----
min_x = repmat(min(X),216,1);
max_x = repmat(max(X),216,1);
X = (X - min_x)./(max_x - min_x);
min_y = repmat(min(Y),216,1);
max_y = repmat(max(Y),216,1);
Y = (Y - min_y)./(max_y - min_y);


warning off;
% disp('����NaiveBayes����');
% O1 = NaiveBayes.fit(XY,L,'distribution','mvmn');% MATLAB �Դ��ĺ��� ̫�� �����Լ�д
[mean1,var1] = My_NaiveBayes(X');%������  (1*216) (1*216)
[mean0,var0] = My_NaiveBayes(Y');%������


% %-------logistic_refression-------------------------------------------
% L1 =  ones( 1,size(X,2) );
% L2 =  zeros( 1,size(Y,2) );
% L = [L1 L2];%��ǩ
% XY = [X,Y]; %����
% 
% min_xy = repmat(min(XY),72,1);
% max_xy = repmat(max(XY),72,1);
% XY = (XY - min_xy)./(max_xy - min_xy);
% 
% global_w = logistic_regression(XY',L');
% w_opt = global_w(2:end);
% w0 = global_w(1);
%%
%----������������ȡ��Ӧ��������������������------
all_samples_f = all_samples( :, patch_index, : );
% all_samples_f = sum( all_samples_f, 2 )./size(patch_index,1);
% all_samples_f = max( all_samples_f,[],2 );% max poolig
% all_samples_f = reshape( all_samples_f, size( all_samples_f, 1), size( all_samples_f, 3) );
all_samples_f  = reshape(all_samples_f,size( all_samples_f, 1)*size( all_samples_f, 2), size( all_samples_f, 3) );%concatenate pooling

% [rate, isclassify]  = fisherTest( all_samples_f, samples_weight, all_samples_flag, w_opt, w0 );%����
[rate, isclassify]  = fisherTest_NaiveBayes( all_samples_f, samples_weight, all_samples_flag, mean1, var1,mean0, var0 );


% ret = struct('w_opt', w_opt, 'w0', w0, 'weight_error_rate', rate, 'feature_indx', feature_indx, 'isclassify', isclassify);%���������
ret = struct('mean1', mean1,'var1',var1,'mean0', mean0,'var0',var0, 'weight_error_rate', rate, 'feature_indx', feature_indx, 'isclassify', isclassify);%���������




end