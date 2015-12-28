
% 从9个patch中，选择3个patch块，所有的组合数
mm = 9;
nn = 3;
SeqSet = combntns( [1:mm],nn );
SeqSet = SeqSet';

id = randperm( size(SeqSet,2), 20 ); %可以从中随机选择部分参与运算
SeqSet = SeqSet( :,id );