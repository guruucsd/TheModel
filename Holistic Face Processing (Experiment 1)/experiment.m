function [] = experiment()

%% set up paths and network constants
dataPath = '/Users/Davis/Desktop/garyComposite/dataset';
dataType = '*.bmp';
numIterations = 100000;

%% run experiment: preprocess data and train data.
preprocessor(dataPath, dataType);
CompositeClassifierTrain(numIterations); % iterations, numHidden

%% experiment complete. data for further testing in mat files
display('experiment complete');
