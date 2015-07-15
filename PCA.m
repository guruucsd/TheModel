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
numPCA = 10;

sigma = dataAllTrain*dataAllTrain'/numImagesTrain; %definition of covariance with zero mean.
[eigenspace, eigenval, V] = svd(sigma); %we find the eigenspace and the eigenvalues

PCATrainData = [];
PCATestData = [];
PCAValidData = [];

for i = 1:numPCA
    data_rot = eigenspace(:,i)' * dataAllTrain;
    PCATrainData = [PCATrainData; data_rot]; %check the dimensionality of this thing.
end

for i = 1:numPCA
    data_rot = eigenspace(:,i)' * dataAllTest;
    PCATestData = [PCATestData; data_rot]; %check the dimensionality of this thing.
end

for i = 1:numPCA
    data_rot = eigenspace(:,i)' * dataAllValid;
    PCAValidData = [PCAValidData; data_rot]; %check the dimensionality of this thing.
end




fprintf('    PCA extraction complete. \n');



