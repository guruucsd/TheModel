function [theta_out_DATASET testPerformanceResult]=NetworkTrainingFaceExpert(faceData, numHidden,nIterFace)

%   [theta_out_DATASET testPerformanceResult hidden_activation_PCA_all]=NetworkTrainingFaceExpert(faceData, numHidden,nIterFace)
%   Network training for faces. 
%   Author: Panqu Wang
%   This is only a toy version. Do not distribute without permission.
%   Parameters:
%   Input:
%   faceData:   training set and test set containing faces
%   numHidden:  number of hidden units in the neural network
%   nIterFace:  number of training iterations for face expert network.
%   Output:
%   theta_out_DATASET:      weights (if further analysis needed)
%   testPerformanceResult:  face recognition result on test set


trainingSet=faceData.trainingSet(1:end);
testSet=faceData.testSet(1:end);


%% initializing.
num_output=size(trainingSet,2);
input_size=size(trainingSet{1,1},1);
num_train=size(trainingSet{1,1},2);
num_hidden=numHidden;

%weights
W_in_hd1=-0.5+rand(num_hidden,input_size);
W_hd_op1=-0.5+rand(num_output,num_hidden);

theta = [W_in_hd1(:); W_hd_op1(:)];

lambda=0.015; % learning rate
threshold=0.005; % threshold for errors
momentum=0.01;
% mu=0.001; % regularization strength, not used here
error_avg=1;
epoch=1;
theta_old=theta;
theta_new=theta;

%% training
while error_avg>threshold & epoch<=nIterFace

    %input node
    for i=1:num_output %number of output nodes
        for j=1:num_train %number of training images
        node=struct;
        node.id=num_train*(i-1)+j;

        %which dataset to use? Here use full spatial frequency to train
        node.val=trainingSet{i}(:,j);    
        node.type='input';
        node.obj=i;
        node.dsoutput=zeros(num_output,1);
        node.dsoutput(i)=1;
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
        [in_hd1 hd_op1]=GetParameterMonoNetwork(total_gradients,input_size,num_output,num_train,num_hidden);
        error_store(num_node)=errors;
        chg_in_hd1=lambda*in_hd1;
        chg_hd_op1=lambda*hd_op1;
        theta_new=theta-[chg_in_hd1(:); chg_hd_op1(:);]+momentum*(theta-theta_old);
        theta_old=theta;
        theta=theta_new;
    end
    save('theta_temp','theta');

    % uncomment this to get the time vs nodes value figure
    % record the performance (accuracy) and error vs. training epochs.
    if mod(epoch,1)==0
       [ave_acc_train(epoch) ave_acc_test(epoch) ...
           mean_error_trainingset(epoch) mean_error_testset(epoch)]=NetworkTestingFaceExpertTemp(epoch,faceData,theta,num_hidden);
    end
    display(['Training... epoch=' num2str(epoch) ' test accuracy=' num2str(ave_acc_test(epoch)) ' training accuracy=' num2str(ave_acc_train(epoch)) ' mean_error_trainingset=' num2str(mean_error_trainingset(epoch)) ])

    error_avg=mean(error_store);
    epoch=epoch+1;
end
theta_out_DATASET=theta;

testPerformanceResult=mean(ave_acc_test(end),1);

save('theta_result_after_face_expert_training','theta_out_DATASET')
save('result_face_mono','ave_acc_train','ave_acc_test','mean_error_trainingset','mean_error_testset')

end
