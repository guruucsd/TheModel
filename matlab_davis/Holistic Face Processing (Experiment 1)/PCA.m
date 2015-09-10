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


%subtract out the mean for every image.
inlen = size(dataAllTrain,1);
avg = sum(dataAllTrain')/numImagesTrain; %average of each dimension
dataAllTrain = bsxfun(@minus,dataAllTrain',avg);
dataAllTest = bsxfun(@minus,dataAllTest',avg);
dataAllValid = bsxfun(@minus,dataAllValid',avg);

%n images m dimensions, nxm data matrix.

%% calculate the covariance matrix and find it's eigenvalues/vectors

numPCA = 20;

%Kohonen and Lowe Trick for small covariance matrix.
display('    constructing covariance matrix...');
sigma = dataAllTrain*dataAllTrain'/(numImagesTrain-1); 
display('    extracting eigendata...');
[eigenspace_T, eigenval_T] = svd(sigma); %we find the eigenspace and the eigenvalues

%transform eigenvalues and eigenspace back into non-transposed forms

%normalize eigenspace
eigenspace = dataAllTrain'*eigenspace_T;


%get n PC's
PCATrainData = [];
PCATestData = [];
PCAValidData = [];

% construct then normalize the eigenvectors.

data_rot_train = dataAllTrain*eigenspace;
data_rot_valid = dataAllValid*eigenspace;
data_rot_test = dataAllTest*eigenspace;

snorm = diag(sqrt(data_rot_train'*data_rot_train));
data_rot_train = data_rot_train/diag(snorm);
data_rot_test = data_rot_test/diag(snorm);
data_rot_valid = data_rot_valid/diag(snorm);

fprintf('     extracting principal components...')
PCATrainData = data_rot_train(:,1:numPCA);
PCAValidData = data_rot_valid(:,1:numPCA);
PCATestData = data_rot_test(:,1:numPCA);



%{
dataAll = [dataAllTrain,dataAllTest];
pcaDat = pca(dataAll)';

comp = 10;

PCATrainData_matlab = pcaDat(1:comp,1:numImagesTrain);
PCATestData_matlab = pcaDat(1:comp,numImagesTrain+1:numImagesTrain+numImagesTest);
PCAValidData_matlab = PCATrainData;

%}

PCATrainData = PCATrainData';
PCATestData = PCATestData';
PCAValidData = PCAValidData';

fprintf('    PCA extraction complete. \n');



