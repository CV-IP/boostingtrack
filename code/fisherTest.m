%每一列都是一个样本.
%返回错误率.
%%
function [rate, isclassify]  = fisherTest( samples, samples_weight,  flags, w_opt, w0 )
    rate = 0;
    isclassify = ones(1, length(flags));
    value = w_opt' * samples + w0;
    b = value .* flags;
    c = b<0;
    rate = sum ( samples_weight(c) );
    isclassify( c ) = 0;
    
%     for i_indx = 1:length(flags)
%         value = w_opt' * samples(:, i_indx) + w0;     
%         if value * flags(i_indx) < 0
%             rate = rate + samples_weight(i_indx);      
%             isclassify(i_indx) = 0;
%         end   
%     end
        
    


 %-----------logistic--------------------------------
%     rate = 0;
%     isclassify = ones(1, length(flags));
%     value = -(w_opt' * samples + w0);
% %     value = -value;
%     value = 1./(1+exp(value));
% 
%     b = value .* flags;
%     c = b<0;
%     rate = sum ( samples_weight(c) );
%     isclassify( c ) = 0;   
end