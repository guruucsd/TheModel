function []=RunExperiment()

%   RunExperiment()
%   Train the network using TM ("The Model", Dailey and Cottrell (1999))
%   using the data from PreprocessDataSet.m
%   Author: Panqu Wang
%   This is only a toy version. Do not distribute without permission.
%   12 training images, 4 testing images per individual.


close all; clear all; clc;
% load SamplePreprocessedData % get preprocessedData (12+4 each)
load PreprocessedData

% find index of faces (not necessary in this sample code)
for indexPreprocessedData=1:length(preprocessedData)
    name=preprocessedData(indexPreprocessedData).name;
    if strcmp(name,'Faces')
       faceDataIndex=indexPreprocessedData;
    end
end
        

% Train the face expert network
nIterFace=100;
numHidden=8;
learning_rate=0.015; % learning rate
error_threshold=0.005; % threshold for errors
momentum=0.01;
[weightFaceExpertNetwork testPerformanceFace]=NetworkTrainingFaceExpert(preprocessedData(faceDataIndex),numHidden,nIterFace,learning_rate,error_threshold, momentum);           

display(['Expert Network Training Finished. Test Performance on faces=' num2str(testPerformanceFace)]);

