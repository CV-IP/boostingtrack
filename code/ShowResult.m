function ShowResult( image_color, affpara, i ,template_size )
%image_color : 原图，未经转换
%affpara : 最佳仿射参数，未转换
%i : 当前是第几帧
%% 
%     if(i==1)
%         figure('position',[ 100 100 size(image_color,2) size(image_color,1) ]);
%         set(gcf,'DoubleBuffer','on','MenuBar','none');
%     end
%     
%     axes(axes('position', [0 0 1.0 1.0]));
%     imagesc(image_color, [0,1]);
%     numStr = sprintf('#%03d', i);
%     text(10,20,numStr,'Color','r', 'FontWeight','bold', 'FontSize',20);
    
%     color = [ 1 0 0 ];
set (gcf,'Position',[400,200,size(image_color,2),size(image_color,1)]);
imshow(image_color,'border','tight','InitialMagnification','fit');
hold on;
drawbox(template_size, affparam2mat(affpara), 'Color', 'y', 'LineWidth', 2);
text(10,50,int2str(i),'color','red','fontsize',20);
hold off;

end

