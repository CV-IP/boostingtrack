
% ��9��patch�У�ѡ��3��patch�飬���е������
mm = 9;
nn = 3;
SeqSet = combntns( [1:mm],nn );
SeqSet = SeqSet';

id = randperm( size(SeqSet,2), 20 ); %���Դ������ѡ�񲿷ֲ�������
SeqSet = SeqSet( :,id );