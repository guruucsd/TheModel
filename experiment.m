function [] = experiment()
clear all; close all;
%% set up paths and network constants
dataPath = '/Users/Davis/Desktop/garyComposite/dataset';
dataType = '*.jpg';
numIterations = 10;
numHidden = 50;

%% run experiment: preprocess data and train data.
preprocessor(dataPath, dataType);
CompositeClassifierTrain(numIterations, numHidden); % iterations, numHidden

%% experiment complete. data for further testing in mat files
display('experiment complete');
