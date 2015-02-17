function [theta_out_DATASET testPerformanceResult testPerformanceFaceInExpertResult hidden_activation_PCA_all]=NetworkTrainingNewExpertCottrellSuggest(allExpertData, newExpertName, numHidden,nIterNewExpert,weightFaceExpertNetwork)

%   [theta_out_DATASET testPerformanceResult hidden_activation_PCA_all]=NetworkTrainingNewExpert(allExpertData, newExpertName, numHidden,nIterNewExpert,weightFaceExpertNetwork)
%   Network training for both face and non-face categories. 
%
%   Parameters:
%   Input:
%   allExpertData:           training set and test set
%   newExpertName:           the name of new category
%   numHidden:               number of hidden units
%   nIterNewExpert:          number of training iterations for this mixed-expert network
%   weightFaceExpertNetwork: the saved weights for the pretrained face network
%
%   Output:
%   theta_out_DATASET:                 weights (if further analysis needed)
%   testPerformanceResult:             object recognition result on test set
%   testPerformanceFaceInExpertResult: face recognition result on test set
%   hidden_activation_PCA_all:         hidden unit activations for mixed-experts
%
%   Author: Panqu Wang
%   This is only a toy version. Do not distribute without permission.

for indexPreprocessedData=1:length(allExpertData)
    name=allExpertData(indexPreprocessedData).name;
    if strcmp(name,'Faces')
       faceDataIndex=indexPreprocessedData;
    elseif strcmp(name,newExpertName)
       newExpertDataIndex=indexPreprocessedData;
    end
end

trainingSetNewExpert=allExpertData(newExpertDataIndex).trainingSet;
testSetNewExpert=allExpertData(newExpertDataIndex).testSet;

trainingSetFaceExpert=allExpertData(faceDataIndex).trainingSet;
testSetFaceExpert=allExpertData(faceDataIndex).testSet;

runtime=1;

%% initializing.
% increase the number of output nodes
num_output_face=size(trainingSetFaceExpert,2);
num_output=size(trainingSetFaceExpert,2)+size(trainingSetNewExpert,2);
num_output_new_expert=num_output-num_output_face;
num_dataset=size(trainingSetNewExpert,1);
input_size=size(trainingSetNewExpert{1,1},1);
input_size_face=size(trainingSetFaceExpert{1,1},1);
num_train=size(trainingSetNewExpert{1,1},2);
num_train_face=size(trainingSetFaceExpert{1,1},2);
num_hidden=numHidden;
num_hidden_face=numHidden;
[in_hd1_face hd_op1_face]=GetParameterMonoNetwork(weightFaceExpertNetwork,input_size_face,num_output_face,num_train_face,num_hidden_face);

% weights initialization
W_in_hd1=in_hd1_face;
W_hd_op1=[hd_op1_face;rand(num_output-num_output_face,num_hidden)];

theta = [W_in_hd1(:); W_hd_op1(:)];

lambda=0.015; % learning rate
threshold=0.005; % threshold for errors
momentum=0.01;
error_avg=1;
epoch=1;
theta_old=theta;
theta_new=theta;

%% training
while error_avg>threshold & epoch<=nIterNewExpert


    %input value        
    for i=1:num_output %number of output nodes for new expert
        for j=1:num_train %number of training images
        node=struct;
        node.id=num_train*(i-1)+j;
        if i>num_output_face
            node.val=trainingSetNewExpert{i-num_output_face}(:,j);
        else
            node.val=trainingSetFaceExpert{i}(:,j);
        end

        node.type='input';
        node.obj=i;
        node.dsoutput=zeros(num_output,1); % now we have 2X number of output nodes
        node.dsoutput(i)=1;
        node.color=nodecolor(newExpertName,node.obj);
        nodes_temp{num_train*(i-1)+j}=node;
        end
    end


    nodes=cell(1,size(nodes_temp,2));
    order=randperm(length(nodes));

    % randomize the nodes
    for order_count=1:length(nodes)
        nodes{order_count}=nodes_temp{order(order_count)};    
    end

    for num_node=1:length(nodes)    
        [errors total_gradients hidden_activation]=backprop_dataset_hidden_sgd_mono(theta_new,nodes{num_node},input_size,num_output,num_train,num_hidden);

        % Manipulating the selection of hidden units activations
        if nargout>3
            if epoch==1       
                hidden_activation_PCA_all{1}.hidden_activation_PCA(:,num_node)=hidden_activation;
                hidden_activation_PCA_all{1}.name=newExpertName;
                hidden_activation_PCA_all{1}.nodeid_PCA(num_node)=nodes{num_node}.obj;    %which object is it?
                hidden_activation_PCA_all{1}.color_PCA(num_node,:)=nodes{num_node}.color; %which color should it be in RGB format?
            end

            if epoch==5
                hidden_activation_PCA_all{2}.hidden_activation_PCA(:,num_node)=hidden_activation;
                hidden_activation_PCA_all{2}.name=newExpertName;
                hidden_activation_PCA_all{2}.nodeid_PCA(num_node)=nodes{num_node}.obj;    %which object is it?
                hidden_activation_PCA_all{2}.color_PCA(num_node,:)=nodes{num_node}.color; %which color should it be in RGB format?
            end

            if epoch==nIterNewExpert
                hidden_activation_PCA_all{3}.hidden_activation_PCA(:,num_node)=hidden_activation;
                hidden_activation_PCA_all{3}.name=newExpertName;
                hidden_activation_PCA_all{3}.nodeid_PCA(num_node)=nodes{num_node}.obj;    %which object is it?
                hidden_activation_PCA_all{3}.color_PCA(num_node,:)=nodes{num_node}.color; %which color should it be in RGB format?
            end
        end

        [in_hd1 hd_op1]=GetParameterMonoNetwork(total_gradients,input_size,num_output,num_train,num_hidden);
        error_store(num_node)=errors;
        chg_in_hd1=lambda*in_hd1;
        chg_hd_op1=lambda*hd_op1;
        theta_new=theta-[chg_in_hd1(:); chg_hd_op1(:);]+momentum*(theta-theta_old);
        theta_old=theta;
        theta=theta_new;
    end
    save('theta_temp','theta');

    % record the performance (accuracy) and error vs. training epochs.
    if mod(epoch,1)==0
       [ave_acc_train(runtime,epoch) ave_acc_train_face_in_expert(runtime,epoch) ave_acc_test(runtime,epoch) ave_acc_test_face_in_expert(runtime,epoch)...
           mean_error_trainingset(runtime,epoch) mean_error_testset(runtime,epoch)]=NetworkTestingNewExpertCottrellSuggestTemp(epoch,allExpertData,newExpertName,theta,num_hidden);
    end

    error_avg=mean(error_store);
    epoch=epoch+1;
end
theta_out_DATASET=theta;


testPerformanceResult=mean(ave_acc_test(end),1);
testPerformanceFaceInExpertResult=mean(ave_acc_test_face_in_expert(end),1);


save('theta_result_after_new_expert_training','theta_out_DATASET')
save('result_expert_mono','ave_acc_train','ave_acc_train_face_in_expert','ave_acc_test','ave_acc_test_face_in_expert','mean_error_trainingset','mean_error_testset')


end
