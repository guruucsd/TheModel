function [PCATrainData, PCATestData, PCAValidData] = PCA(trainSet, testSet, validSet, message)
%our images already have zero mean from the z-scoring.

%% Implement PCA and rotate data.
fprintf('%s \n', message);

%% first, reshape the data to have each column representing one image.

numImagesTrain = size(trainSet,1);
numImagesTest = size(testSet, 1);
numImagesValid = size(validSet, 1);

numScales = size(trainSet,2);  %scales and orientations are same across all sets.
numOrientations = size(trainSet,3);

dataAllTrain = [];
dataAllTest = [];
dataAllValid = [];

display('    reshaping training set...');

for i = 1:numImagesTrain
    dataImTrain = [];
    for s = 1:numScales
        for o = 1:numOrientations
            dataGabTrain = reshape(trainSet{i,s,o}, [size(trainSet{i,s,o},1)*size(trainSet{i,s,o},2),1]);   %feature maps reshaped into columns
            dataImTrain = [dataImTrain; dataGabTrain]; %appending all gabor columns together
        end
    end
    dataAllTrain = [dataAllTrain,dataImTrain]; %each image corresponds to one column
end

display('    reshaping testing set...');

for i = 1:numImagesTest
    dataImTest = [];
    for s = 1:numScales
        for o = 1:numOrientations
            dataGabTest = reshape(testSet{i,s,o}, [size(testSet{i,s,o},1)*size(testSet{i,s,o},2),1]);   %feature maps reshaped into columns
            dataImTest = [dataImTest; dataGabTest]; %appending all gabor columns together
        end
    end
    dataAllTest = [dataAllTest,dataImTest]; %each image corresponds to one column
end

display('    reshaping validation set...');

for i = 1:numImagesValid
dataImValid = [];
    for s = 1:numScales
        for o = 1:numOrientations
            dataGabValid = reshape(validSet{i,s,o}, [size(validSet{i,s,o},1)*size(validSet{i,s,o},2),1]);   %feature maps reshaped into columns
            dataImValid = [dataImValid; dataGabValid]; %appending all gabor columns together
        end
    end
    dataAllValid = [dataAllValid,dataImValid]; %each image corresponds to one column
end



%each col is now an image. the matrix is now n-dataelements x m-images

%% calculate the covariance matrix and find it's eigenvalues/vectors
%{
numPCA = 50;


display('    constructing covariance matrix...');
sigma = dataAllTrain'*dataAllTrain/numImagesTrain; %definition of covariance with zero mean.
display('    extracting eigendata...');
[eigenspace_T, eigenval_T] = svd(sigma); %we find the eigenspace and the eigenvalues

%transform eigenvalues and eigenspace back into non-transposed forms
k = (size(dataAllTrain,2)-1)/(size(dataAllTrain,1)-1);
eigenval  = (1/k)*eigenval_T;
X = dataAllTrain/sqrt(2*k*eigenval);
eigenspace = eigenspace_T*pinv(X);

%get n PC's
PCATrainData = [];
PCATestData = [];
PCAValidData = [];


display('    extracting training set principle components...');
data_rot_train = eigenspace*dataAllTrain;
data_rot_valid = eigenspace*dataAllValid;
data_rot_test = eigenspace*dataAllTest;

for i = 1:numPCA
    fprintf('        %i PC extracted \n', i);
    PCATrainData = [PCATrainData; data_rot_train(i,:)]; %check the dimensionality of this thing.
end

display('    extracting testing set principle components...');
for i = 1:numPCA
    fprintf('        %i PC extracted \n', i);
    PCATestData = [PCATestData; data_rot_test(i,:)]; %check the dimensionality of this thing.
end

display('    extracting validation set principle components...');
for i = 1:numPCA
    fprintf('        %i PC extracted \n', i);
    PCAValidData = [PCAValidData; data_rot_valid(i,:)]; %check the dimensionality of this thing.
end

%}



dataAll = [dataAllTrain,dataAllTest];
pcaDat = pca(dataAll)';

comp = 50;

PCATrainData = pcaDat(1:comp,1:numImagesTrain);
PCATestData = pcaDat(1:comp,numImagesTrain+1:numImagesTrain+numImagesTest);
PCAValidData = PCATrainData;


fprintf('    PCA extraction complete. \n');



