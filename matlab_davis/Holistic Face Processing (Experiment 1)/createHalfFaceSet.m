function [trainSet_half, testSet_half, validSet_half] = createHalfFaceSet(trainSet, testSet, validSet)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    cropPixel = 40;

    numTrainImages = size(trainSet,1);
    numValidImages = size(validSet,1);
    numTestImages = size(testSet, 1);
    
    numSizes = size(trainSet,2);
    numCategories = size(trainSet,3);
    
    imgHeight = size(trainSet{1},1);
    imgWidth = size(trainSet{1},2);

    topHalf = ones(imgHeight, imgWidth);
    bottomHalf = ones(imgHeight, imgWidth);
    
    for i = cropPixel:imgHeight
        topHalf(i,:) = 0.125;
    end
    
    for i = 1:(cropPixel-1)
        bottomHalf(i,:) = 0.125;
    end
    
    trainSet_half = {};
    testSet_half = {};
    validSet_half = {};
    
    
    % training images half
    for i = 1:numTrainImages
        for s = 1:numSizes
            for o = 1:numCategories
                trainSet_half{i,s,o} = trainSet{i,s,o}.*topHalf;
            end
        end 
    end
    
    for i = 1:numTrainImages
        for s = 1:numSizes
            for o = 1:numCategories
                trainSet_half{numTrainImages + i,s,o} = trainSet{i,s,o}.*bottomHalf;
            end
        end
    end
    
    
    %test images half
    for i = 1:numTestImages
        for s = 1:numSizes
            for o = 1:numCategories
                testSet_half{i,s,o} = testSet{i,s,o}.*topHalf;
            end
        end
    end
    
    for i = 1:numTestImages
        for s = 1:numSizes
            for o = 1:numCategories
                testSet_half{numTestImages + i,s,o} = testSet{i,s,o}.*bottomHalf;
            end
        end
    end
    
    
    %valid images half
    for i = 1:numValidImages
        for s = 1:numSizes
            for o = 1:numCategories
                validSet_half{i,s,o} = validSet{i,s,o}.*topHalf;
            end
        end  
    end
    
    for i = 1:numValidImages
        for s = 1:numSizes
            for o = 1:numCategories
                validSet_half{numValidImages + i,s,o} = validSet{i,s,o}.*bottomHalf;
            end
        end
    end
    
    
    
    

end

