% 采样...
% 使用接受拒绝法采样...

function [ samples_indx ] = sample_method( samples_weight, Nnum )
    N = length( samples_weight );
    max_value = max( samples_weight )*1.2;
    samples_indx = zeros( floor(Nnum), 1);
    counter = 0;
    while( counter < Nnum )
        rnd_i = randi([1, N], 1);
        u = rand(1); 
        
        % U <= pi(X) / (M * q(X) )
        if u <= samples_weight(rnd_i) / max_value 
            counter = counter + 1;
            samples_indx( counter ) = rnd_i;    
        end
        
    end
end