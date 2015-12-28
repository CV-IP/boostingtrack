%---检查不合适的粒子
    error_id = ( all_affparam(1,:) - all_affparam(3,:)*template_size(1)/2 ) < 0;
    all_affparam(1,error_id) = all_affparam(3,error_id)*template_size(1)/2;
    error_id = ( all_affparam(1,:) + all_affparam(3,:)*template_size(1)/2 ) > frame_width;
    all_affparam(1,error_id) = frame_width - all_affparam(3,error_id)*template_size(1)/2;
    error_id = ( all_affparam(2,:) - all_affparam(5,:).*all_affparam(3,:)*template_size(1)/2 ) < 0;
    all_affparam(2,error_id) = all_affparam(5,error_id).*all_affparam(3,error_id)*template_size(1)/2;
    error_id = ( all_affparam(2,:) + all_affparam(5,:).*all_affparam(3,:)*template_size(1)/2 ) > frame_height;
    all_affparam(2,error_id) = frame_height - all_affparam(5,error_id).*all_affparam(3,error_id)*template_size(1)/2;