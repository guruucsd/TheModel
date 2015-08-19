function [] = preprocessor(dataPath, dataType)
% this function preprocesses for an entire experiment.
% @author Davis Liang
% @version 1.0
% date: 7/13/15
%% initial setup
numScales = 5;
numOrientations = 8;

trainPath = fullfile(dataPath, 'train');
testPath = fullfile(dataPath, 'test');
validPath = fullfile(dataPath, 'valid');

savePath = '/Users/Davis/Desktop/garyComposite/params/preprocessparams.mat';
savePath2 = '/Users/Davis/Desktop/garyComposite/params/preprocessparams_checkpoint.mat';

%% upload datasets and labels (normalized around zero)
display('loading training set... ');
[trainData, trainLabels] = importNormImages(trainPath, dataType); 
display('loading testing set... ');
[validData, validLabels] = importNormImages(validPath, dataType);
display('loading validation set... ');
[testData, testLabels] = importNormImages(testPath, dataType);


%% create gabors
gabor = createGabors(numScales, numOrientations); %specifically designed for images 256*256
%gabor{s,o} refers to gabor of size s and orientation o
%% filter images and downsample and complex magnitude


testFeature = filterSet(testData, gabor, 'filtering testing set');  %this set smallest ad most variable so filter this first.
validFeature = filterSet(validData, gabor, 'filtering validation set');
trainFeature = filterSet(trainData, gabor, 'filtering training set...');
display('checkpoint. data saved.');

save(savePath2, 'testFeature', 'validFeature', 'trainFeature');

% remember that dataFeature{i,s,o} refers to the feature map corresponding 
%to image i, scale s, and orientation o.

%% additional processing  (z-score across all maps, PCA for n components)
[trainData, testData, validData] = zscore(trainFeature, testFeature, validFeature);
%Z-score by finding the mean and std of the training set, and operating on
%the data sets with these numbers. Z-score across all images.

[trainPCA, testPCA, validPCA] = PCA(trainData, testData, validData, 'extracting PCA for all sets');




save(savePath, 'trainPCA', 'testPCA', 'validPCA', 'trainLabels', 'testLabels', 'validLabels');

display('preprocessing complete. preprocessed data saved.');








