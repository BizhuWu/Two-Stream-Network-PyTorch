UCF101_dir_path = 'Your_Path\UCF-101';
UCF101_dir = dir(UCF101_dir_path);

% dir used to save the RGB images & the optical flow images
RGB_Dir_Path= 'Your_Path\RGB\';
OpticalFlow_Dir_Path= 'Your_Path\OpticalFlow\';

save_interval = 2;

% the first one is '.'; the second one is '..'
for i=3:length(UCF101_dir)
    class_name = UCF101_dir(i).name;
    class_dir_path = strcat(UCF101_dir(i).folder, '\', class_name);
    class_dir = dir(class_dir_path);
    
    RGB_save_class_path = strcat(RGB_Dir_Path, class_name);
    OpticalFlow_save_class_path = strcat(OpticalFlow_Dir_Path, class_name);
    mkdir(RGB_save_class_path);
    mkdir(OpticalFlow_save_class_path);
    
    for j = 3:length(class_dir)
        video_name = class_dir(j).name;
        video_path = strcat(class_dir(j).folder, '\', video_name);
        
        vidReader = VideoReader(video_path);
        vidWidth = vidReader.Width;
        vidHeight = vidReader.Height;
        num_of_frame = vidReader.NumberOfFrames;
        
        vidReader = VideoReader(video_path);
        
        opticFlow = opticalFlowHS;
        
        RGB_frame_save_dir = strcat(RGB_save_class_path, '\', video_name(1: strlength(video_name)-4));
        mkdir(RGB_frame_save_dir);
        OpticalFrame_frame_save_dir = strcat(OpticalFlow_save_class_path, '\', video_name(1: strlength(video_name)-4));
        mkdir(OpticalFrame_frame_save_dir);
        
        i_frame = 1;
        ran_frame_num = randi(num_of_frame);

        while hasFrame(vidReader)
            new_frameRGB = readFrame(vidReader);
            
            if i_frame == ran_frame_num
                % save RGB
                RGB_frame_name = strcat(RGB_frame_save_dir, '\', video_name(1: strlength(video_name)-4), '_', num2str(i_frame));
                imshow(new_frameRGB,'border','tight','initialmagnification','fit');   
                set(gcf,'position',[0 0 vidWidth vidHeight]);
                axis normal;
                saveas(gcf, RGB_frame_name, 'jpg');
            end
            
            
            if mod(i_frame, save_interval) == 0
                %save Optical Flow
                frameGray = rgb2gray(new_frameRGB);
                flow = estimateFlow(opticFlow, frameGray);
                
                x_OpticalFlow_frame_name = strcat(OpticalFrame_frame_save_dir, '\', video_name(1: strlength(video_name)-4), '_', num2str(i_frame), '_x');
                imshow(flow.Vx ,'border','tight','initialmagnification','fit');    
                set(gcf,'position',[0 0 vidWidth vidHeight]);
                axis normal;
                saveas(gcf, x_OpticalFlow_frame_name, 'jpg');
                
                y_OpticalFlow_frame_name = strcat(OpticalFrame_frame_save_dir, '\', video_name(1: strlength(video_name)-4), '_', num2str(i_frame), '_y');
                imshow(flow.Vy ,'border','tight','initialmagnification','fit');    
                set(gcf,'position',[0 0 vidWidth vidHeight]);
                axis normal;
                saveas(gcf, y_OpticalFlow_frame_name, 'jpg');
            end
            
            
            i_frame = i_frame + 1;
        end
    end
end
