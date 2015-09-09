function[] = preprocessDataSetYufeiOriginal()
%Yufei's preprocessing code
clear all;
close all; 
clc;

cats = 7;
trains = 13;
tests = 1;

s = 5; %scale
o = 8; %orientation
im_size = 64; %image size
size_gb=48; %gabor size
k = zeros(1, s); phi = zeros(1, o);
std_dev=pi;
num_pca=8;
dwsp=8;%downsampled rate

%specify some root path
root = '/Users/Davis/Desktop/sample';
%read files from root path%goes to said directory and only shows .BMP images?
files_temp = [dir( fullfile(root,'*.pgm'))];
%initializes empty files struct
files = {};
%adds all files that match parameters given.
for i=1:length(files_temp)
    %if the first letter in the file name is not a ".", gets rid of .dStore
    %stuff I guess.faces

        %adds to fileList
    files{i} = files_temp(i).name;   

end

%for folder 1, provide labels for each.
label = zeros(length(files),1);
for i=1:length(files)
    label(i) = str2num(files{i}(2:3));
end



[label,I] = sort(label);
files = files(I);

curr = 0;
prev = 0;
for i=1:length(label)
    if(prev ~= label(i))
        curr = curr+1;
        prev = label(i);
        label(i) = curr;
    else
        label(i) = curr;
    end
end
count = [1];
for i =2:length(label)
    if label(i) ~= label(i-1)
        count = [count,1];
    else
        count(end) = count(end)+1;
    end
end

train_indice = ones(length(files),1);
start = 1;
for i=1:length(count)
    indice = randperm(count(i));
    temp = 1;%2 testers
    for j =1:temp
        train_indice(indice(j)+start-1) = 0;
    end
    start = start + count(i);
end
label_test = label(train_indice==0);
label_train = label(train_indice==1);

total_train = sum(train_indice);
total_test = length(train_indice) - total_train;
f_filtered_normalized_dwsp_vector_allsub_train = zeros(8*8*s*o, total_train);
f_filtered_normalized_dwsp_vector_allsub_test = zeros(8*8*s*o, total_test);
train_i = 1;
test_i = 1;


%% Gabor filter

for i=1:5
    k(i)=(2*pi/im_size)*2^i;
end
%orientation
for i=1:8
    phi(i)=(pi/8)*(i-1);
end
carrier = zeros(size_gb, size_gb);
envelop = zeros(size_gb, size_gb);
gabor = zeros(size_gb, size_gb, o, s);

for scale=1:size(k,2)
    %figure;
    for orientation=1:size(phi,2)
        for ii=-size_gb+1:size_gb
            for j=-size_gb+1:size_gb
                carrier(ii+size_gb,j+size_gb)=exp(1i*(k(scale)*cos(phi(orientation))*ii+k(scale)*sin(phi(orientation))*j));
                envelop(ii+size_gb,j+size_gb)=exp(-(k(scale)^2*(ii^2+j^2))/(2*std_dev*std_dev));
                gabor(ii+size_gb,j+size_gb,orientation,scale)=carrier(ii+size_gb,j+size_gb)*envelop(ii+size_gb,j+size_gb);
            end
        end
                           %subplot(2,4,orientation); imshow(gabor(:,:,orientation,scale),[]);

    end
end

%%


for i=1:length(files)
    display(['Image ' num2str(i)]);
    file_path = fullfile(root, files{i});

    %% for each image, do gabor_filtering
    f=imread(file_path);
    if size(size(f),2)==2
        f=imresize(im2double(f),[im_size im_size]);
    else
        f=rgb2gray(im2double(f));
        f=imresize(f,[im_size im_size]);
    end
    %constructing gabor filter (16*16) and filtering the input image
    for scale=1:size(k,2)
        %scale
        for orientation=1:size(phi,2)
            %subplot(2,4,orientation); imshow(gabor(:,:,orientation),[]);
            f_filtered{scale}(:,:,orientation)=imfilter(f,gabor(:,:,orientation, scale),'replicate','conv');       
            f_filtered_dwsp{scale}(:,:,orientation)=imresize(imfilter(f,gabor(:,:,orientation,scale),'replicate','conv'),[dwsp,dwsp]);       
            %figure;
            %imshow(f_filtered{scale}(:,:,orientation),[])
        end
        %now we have 8 orientations for each scale, downsample and do normalization
        for orientation=1:size(phi,2)
            f_filtered_normalized_dwsp{scale}(:,:,orientation)=abs(f_filtered_dwsp{scale}(:,:,orientation));               
        end
    end
    %normalize them for each scale
    for scale=1:size(k,2);
        f_filtered_normalized_dwsp_vector((scale-1)*dwsp*dwsp*size(phi,2)+1:(scale)*dwsp*dwsp*size(phi,2))=f_filtered_normalized_dwsp{scale}(:);             
    end
    %testset and trainingset assembly
    if train_indice(i) == 0
        f_filtered_normalized_dwsp_vector_allsub_test(:,test_i)=f_filtered_normalized_dwsp_vector(:);
        test_i = test_i + 1;
        %f_filtered_normalized_dwsp_vector_allsub_test(:,end+1)=f_filtered_normalized_dwsp_vector;
    else
        f_filtered_normalized_dwsp_vector_allsub_train(:,train_i)=f_filtered_normalized_dwsp_vector(:);
        %f_filtered_normalized_dwsp_vector_allsub_train(:,end+1)=f_filtered_normalized_dwsp_vector;
        train_i = train_i + 1;
    end
    clear f_filtered_normalized_dwsp_vector;
end

%%%%%%%%%%%z-score%%%%%%%%%%%
s=std(f_filtered_normalized_dwsp_vector_allsub_train,[],2);
m=mean(f_filtered_normalized_dwsp_vector_allsub_train,2);
f_filtered_normalized_dwsp_vector_allsub_train=bsxfun(@minus, f_filtered_normalized_dwsp_vector_allsub_train, m);
f_filtered_normalized_dwsp_vector_allsub_train=bsxfun(@rdivide, f_filtered_normalized_dwsp_vector_allsub_train, s+.1);
f_filtered_normalized_dwsp_vector_allsub_test=bsxfun(@minus, f_filtered_normalized_dwsp_vector_allsub_test, m);
f_filtered_normalized_dwsp_vector_allsub_test=bsxfun(@rdivide, f_filtered_normalized_dwsp_vector_allsub_test, s+.1);
%%%%%%%%%%%z-score%%%%%%%%%%%


data.train = f_filtered_normalized_dwsp_vector_allsub_train;
data.test = f_filtered_normalized_dwsp_vector_allsub_test;
data.train_label = label_train;
data.test_label = label_test;
a = 'pofa_zscore_GaborData_identity.mat';
save(a,'data');

    %PCA on different scale
    
    
    
    
for scales=1:size(k,2) 
    scale_all=f_filtered_normalized_dwsp_vector_allsub_train((scales-1)*dwsp*dwsp*size(phi,2)+1:scales*dwsp*dwsp*size(phi,2),:);
    scale_all_test=f_filtered_normalized_dwsp_vector_allsub_test((scales-1)*dwsp*dwsp*size(phi,2)+1:scales*dwsp*dwsp*size(phi,2),:);
    mean_images=mean(scale_all,2);    
        
    %turk and pentland trick, for each scale
    mean_subst=scale_all-repmat(mean_images,1,total_train);
    mean_subst_test=scale_all_test-repmat(mean_images,1,total_test);
    cov_scale=(mean_subst'*mean_subst)*(1/total_train); %(estimate of covariance)
    [vector_temp, value]=eig(cov_scale);
    vector_biggest=vector_temp(:,end-num_pca+1:end);
        
    %original principal components
    vector_ori=mean_subst*vector_biggest;
    vector_ori = normc(vector_ori);
    %projection onto the basis vector vector_ori(dimension 512-dimension 8)

    
    %normal
    %%%%%z-score%%%%%
    temp = vector_ori'*mean_subst;
    s=std(temp,[],2);
    m=mean(temp,2);
    temp=bsxfun(@minus, temp, m);
    f_PCA_scale_normal=bsxfun(@rdivide, temp, s);
    temp = vector_ori'*mean_subst_test;
    temp=bsxfun(@minus, temp, m);
    f_PCA_scale_test_normal=bsxfun(@rdivide, temp, s);
    %%%%%%%%%%%%%%%%%%%%

%         f_PCA_scale_normal=(vector_ori(:,:,scales)'*mean_subst(:,:,scales));
%         f_PCA_scale_test_normal=(vector_ori(:,:,scales)'*mean_subst_test(:,:,scales));        
        
    f_PCA_temp_normal((scales-1)*num_pca+1:scales*num_pca,:)=f_PCA_scale_normal;
    f_PCA_test_temp_normal((scales-1)*num_pca+1:scales*num_pca,:)=f_PCA_scale_test_normal;
end


data.train = f_PCA_temp_normal;
data.test = f_PCA_test_temp_normal;



%SETUP OrganizedDataSet
organizedData.train = [];
organizedData.test = [];
counter = 0;
for folders=1:cats
    for images=1:trains
        organizedData.train{folders}(:,images) = data.train(:,images+(counter*trains));
        organizedData.trainlabel{folders}(images) = data.train_label(images*counter);
    end

    
    for images=1:tests
        organizedData.test{folders}(:,images) = data.test(:,images+(counter*tests));
        organizedData.testlabel{folders}(images) = data.test_label(images*counter);
    end
    counter = counter + 1;
end


%a = strcat('pofa_zscore_PreprocessedData_identity_',int2str(num_pca),'.mat');
%save(a,'data')
a = strcat('preDat2', '.mat');
save(a,'organizedData');

%train_image = files(train_indice == 1);
%test_image = files(train_indice == 0);

%information.train= train_image;
%information.test = test_image;



%a = strcat('pofa_zscore_identity_info.mat');
%save(a,'information');

