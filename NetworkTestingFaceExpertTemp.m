function [ave_acc_train_expert ave_acc_test_expert  mean_error_trainingset mean_error_testset]=NetworkTestingFaceExpertTemp(epoch,faceData,theta,numHidden)

trainingSet=faceData.trainingSet(1:end);
testSet=faceData.testSet(1:end);

num_output=size(trainingSet,2);
num_dataset=size(trainingSet,1);
input_size=size(trainingSet{1,1},1);
num_train=size(trainingSet{1,1},2);
num_hidden=numHidden;

[W_in_hd1 W_hd_op1]=GetParameterMonoNetwork(theta,input_size,num_output,num_train,num_hidden);

%test set, for each class
accuracy_test=zeros(size(testSet,2),1);
inaccuracy_test=zeros(size(testSet,2),1);
for i=1:num_output
    class=i;
    for j=1:size(testSet{i},2)
        node=struct;
        node.val=testSet{i}(:,j);        
        node.obj=i;
        node.dsoutput=zeros(num_output,1);
        node.dsoutput(i)=1;
        output=W_hd_op1*sigmoid(W_in_hd1*node.val);
        
        error_testset((i-1)*(size(testSet{i},2))+j)=sum((node.dsoutput-output).^2);
        
        index=find(output==max(output));
        if index==node.obj;
            accuracy_test(i)=accuracy_test(i)+1;
        else
            inaccuracy_test(i)=inaccuracy_test(i)+1;
        end
    end
end
total_accuracy_test=accuracy_test./(accuracy_test+inaccuracy_test);
ave_acc_test_expert=mean(total_accuracy_test(1:end));
mean_error_testset=mean(error_testset);

% training set
accuracy_train=zeros(size(trainingSet,2),1);
inaccuracy_train=zeros(size(trainingSet,2),1);
 for i=1:num_output
    class=i;
    for j=1:num_train
        node=struct;
        %% full freq training set
        node.val=trainingSet{i}(:,j);
        node.obj=i;
        node.dsoutput=zeros(num_output,1);
        node.dsoutput(i)=1;
        
        output=W_hd_op1*sigmoid(W_in_hd1*node.val);
        error_trainingset((i-1)*num_train+j)=sum((node.dsoutput-output).^2);
                
        index=find(output==max(output));
        if index==node.obj;
            accuracy_train(i)=accuracy_train(i)+1;
        else
            inaccuracy_train(i)=inaccuracy_train(i)+1;
        end
    end
 end
total_accuracy_train=accuracy_train./(accuracy_train+inaccuracy_train);
ave_acc_train_expert=mean(total_accuracy_train(1:end));
mean_error_trainingset=mean(error_trainingset);

end
