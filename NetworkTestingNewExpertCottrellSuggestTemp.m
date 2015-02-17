function [ave_acc_train_expert ave_acc_train_face_in_expert ave_acc_test_expert ave_acc_test_face_in_expert  mean_error_trainingset mean_error_testset]=NetworkTestingNewExpertCottrellSuggestTemp(epoch,allExpertData,newExpertName,theta,numHidden)

%   Measure the object recognition accuracy on both training and test set.
%   Author: Panqu Wang

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

[W_in_hd1 W_hd_op1]=GetParameterMonoNetwork(theta,input_size,num_output,num_train,num_hidden);


%test set, for each class
accuracy_test=zeros(size(testSetNewExpert,2)+size(testSetFaceExpert,2),1);
inaccuracy_test=zeros(size(testSetNewExpert,2)+size(testSetFaceExpert,2),1);

for i=1:num_output
    class=i;
    for j=1:4
        node=struct;
        %% full freq test set
        if i>num_output_face
        node.val=testSetNewExpert{i-num_output_face}(:,j);
        node.lfp=testSetNewExpert{i-num_output_face}(:,j);
        node.hfp=testSetNewExpert{i-num_output_face}(:,j);
        else
        node.val=testSetFaceExpert{i}(:,j); 
        node.lfp=testSetFaceExpert{i}(:,j);
        node.hfp=testSetFaceExpert{i}(:,j);
        end
            
        node.obj=i;
        node.dsoutput=zeros(num_output,1);
        node.dsoutput(i)=1;
        
        output=W_hd_op1*sigmoid(W_in_hd1*node.val);
        error_testset((i-1)*(4)+j)=sum((node.dsoutput(1:end)-output(1:end)).^2);
        
        index=find(output==max(output));
        if index==node.obj;
            accuracy_test(i)=accuracy_test(i)+1;
        else
            inaccuracy_test(i)=inaccuracy_test(i)+1;
        end
    end
end
total_accuracy_test=accuracy_test./(accuracy_test+inaccuracy_test);
ave_acc_test_expert=mean(total_accuracy_test(num_output_face+1:end));
ave_acc_test_face_in_expert=mean(total_accuracy_test(1:num_output_face));

mean_error_testset=mean(error_testset);





% training set
accuracy_train=zeros(size(trainingSetNewExpert,2)+size(trainingSetFaceExpert,2),1);
inaccuracy_train=zeros(size(trainingSetNewExpert,2)+size(trainingSetFaceExpert,2),1);
 for i=1:num_output
    class=i;
    for j=1:num_train
        node=struct;
        %% full freq training set
        if i>num_output_face
        node.val=trainingSetNewExpert{i-num_output_face}(:,j);
        node.lfp=trainingSetNewExpert{i-num_output_face}(:,j);
        node.hfp=trainingSetNewExpert{i-num_output_face}(:,j);            
        else
        node.val=trainingSetFaceExpert{i}(:,j);
        node.lfp=trainingSetFaceExpert{i}(:,j);
        node.hfp=trainingSetFaceExpert{i}(:,j); 
        end
            
        node.obj=i;
        node.dsoutput=zeros(num_output,1);
        node.dsoutput(i)=1;
        
        output=W_hd_op1*sigmoid(W_in_hd1*node.val);
        error_trainingset((i-1)*num_train+j)=sum((node.dsoutput(1:end)-output(1:end)).^2);
                
        index=find(output==max(output));
        if index==node.obj;
            accuracy_train(i)=accuracy_train(i)+1;
        else
            inaccuracy_train(i)=inaccuracy_train(i)+1;
        end
    end
 end
total_accuracy_train=accuracy_train./(accuracy_train+inaccuracy_train);
ave_acc_train_expert=mean(total_accuracy_train(num_output_face+1:end));
ave_acc_train_face_in_expert=mean(total_accuracy_train(1:num_output_face));

mean_error_trainingset=mean(error_trainingset);

end
