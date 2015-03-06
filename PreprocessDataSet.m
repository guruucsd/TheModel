function [] = PreprocessDataSet()

%   PreprocessDataSet()
%   Training and test set assembly for training TM ("The Model", Dailey and Cottrell (1999))
%   Author: Panqu Wang
%   This is only a toy version. Do not distribute without permission.
%   12 training images, 4 testing images per individual.


% Finding location of data set.
loc=['./SampleDataSet'];
cd(loc);
objectparent=dir(pwd);
preprocessedData=[];

for objectname=3:length(objectparent)
    name=objectparent(objectname).name;
    cd(name)
    object=dir(pwd);
    
    num_each=16;
    num_test=4;
    total_num=(length(object)-2)*num_each;
    total_train=(length(object)-2)*(num_each-num_test);
    total_test=(length(object)-2)*(num_test);
    num_pca=8;
    size_gb=48;
    std_dev=pi;
    dwsp=8;%downsampled rate
    
    display 'Start Preprocessing...'

    %% for each subject (Tom, Mary, ...) in given object (faces, ...)
    for i=3:length(object)
        display(['Subject ' num2str(i-2)]);
        cd(object(i).name);
        subject=dir(pwd);
        nimage(i-2)=length(subject)-2;%number of training image
        order{i-2}=1:nimage(i-2);

        %% for each image in given subject, do gabor_filtering
        trainIndex=1;
        for current_number=1:num_each
            f=imread(subject(order{i-2}(current_number)+2).name);
            if size(size(f),2)==2
                f=imresize(im2double(f),[64 64]);
            else
                f=rgb2gray(im2double(f)); 
                f=imresize(f,[64 64]);
            end
            [height width]=size(f); 
            %scale
            for temp=1:5
                k(temp)=(2*pi/width)*2^temp;
            end
            %orientation
            for temp=1:8
                phi(temp)=(pi/8)*(temp-1);
            end
            %constructing gabor filter (16*16) and filtering the input image
            for scale=1:size(k,2)
            %scale
                for orientation=1:size(phi,2)
                    for ii=-size_gb+1:size_gb
                        for j=-size_gb+1:size_gb
                            carrier(ii+size_gb,j+size_gb)=exp(1i*(k(scale)*cos(phi(orientation))*ii+k(scale)*sin(phi(orientation))*j));
                            envelop(ii+size_gb,j+size_gb)=exp(-(k(scale)^2*(ii^2+j^2))/(2*std_dev*std_dev));
                            gabor(ii+size_gb,j+size_gb,orientation)=carrier(ii+size_gb,j+size_gb)*envelop(ii+size_gb,j+size_gb);
                        end
                    end
%                     subplot(2,4,orientation); imshow(gabor(:,:,orientation),[]);
                    f_filtered{scale}(:,:,orientation)=imfilter(f,gabor(:,:,orientation),'replicate','conv');       
%                     imshow(f_filtered{scale}(:,:,orientation),[])
                end
                
                %now we have 8 orientations for each scale, downsample and do normalization
                for orientation=1:size(phi,2)
                   f_filtered_dwsp{scale}(:,:,orientation)=imresize(f_filtered{scale}(:,:,orientation),[dwsp dwsp]);
                   f_filtered_normalized_dwsp{scale}(:,:,orientation)=abs(f_filtered_dwsp{scale}(:,:,orientation))./sum(abs(f_filtered_dwsp{scale}),3);               
                end
            end
            for scale=1:size(k,2);
                    f_filtered_normalized_dwsp_vector((scale-1)*dwsp*dwsp*size(phi,2)+1:(scale)*dwsp*dwsp*size(phi,2),current_number)=f_filtered_normalized_dwsp{scale}(:);             
            end      
            
            %testset and trainingset assembly
            if mod(current_number,4)==0
                f_filtered_normalized_dwsp_vector_allsub_test(:,(i-3)*num_test+(current_number)/4)=f_filtered_normalized_dwsp_vector(:,current_number);
            else
                f_filtered_normalized_dwsp_vector_allsub_train(:,(i-3)*(num_each-num_test)+trainIndex)=f_filtered_normalized_dwsp_vector(:,current_number);
                trainIndex=trainIndex+1;
            end
        end
        clear f_filtered_normalized_dwsp_vector;
        cd ..
    end

    %PCA on different scale
    for scales=1:size(k,2) 
        scale_all(:,:,scales)=f_filtered_normalized_dwsp_vector_allsub_train((scales-1)*dwsp*dwsp*size(phi,2)+1:scales*dwsp*dwsp*size(phi,2),:);
        scale_all_test(:,:,scales)=f_filtered_normalized_dwsp_vector_allsub_test((scales-1)*dwsp*dwsp*size(phi,2)+1:scales*dwsp*dwsp*size(phi,2),:);
        mean_images(:,scales)=mean(scale_all(:,:,scales),2);    
        
        %turk and pentland trick, for each scale
        mean_subst(:,:,scales)=scale_all(:,:,scales)-repmat(mean_images(:,scales),1,total_train);
        mean_subst_test(:,:,scales)=scale_all_test(:,:,scales)-repmat(mean_images(:,scales),1,total_test);
        cov_scale=(mean_subst(:,:,scales)'*mean_subst(:,:,scales))*(1/total_train); %(estimate of covariance)
        [vector_temp value]=eig(cov_scale);
        vector_biggest=vector_temp(:,end-num_pca+1:end);
        
        %original principal components
        vector_ori(:,:,scales)=mean_subst(:,:,scales)*vector_biggest;

        %projection onto the basis vector vector_ori(dimension 512-dimension 8)
        %normal
        f_PCA_scale_normal=zscore(vector_ori(:,:,scales)'*mean_subst(:,:,scales));
        f_PCA_scale_test_normal=zscore(vector_ori(:,:,scales)'*mean_subst_test(:,:,scales));
        f_PCA_temp_normal((scales-1)*num_pca+1:scales*num_pca,:)=f_PCA_scale_normal;
        f_PCA_test_temp_normal((scales-1)*num_pca+1:scales*num_pca,:)=f_PCA_scale_test_normal;
      
    end
    
    for i=1:length(object)-2
        f_PCA_normal_DATASET_train{i}=f_PCA_temp_normal(:,(i-1)*(num_each-num_test)+1:i*(num_each-num_test));
        f_PCA_normal_DATASET_test{i}=f_PCA_test_temp_normal(:,(i-1)*num_test+1:i*num_test);  
    end
    
    preprocessedData(objectname-2).name=name;
    preprocessedData(objectname-2).trainingSet=f_PCA_normal_DATASET_train;
    preprocessedData(objectname-2).testSet=f_PCA_normal_DATASET_test;
    cd ..
end
    display 'finished.'
    save(['../SamplePreprocessedData.mat'],'preprocessedData')
    cd ../..
end

