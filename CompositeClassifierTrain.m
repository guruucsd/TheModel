function[] = CompositeClassifierTrain(numIter)



savePath = '/Users/Davis/Desktop/garyComposite/params/networkparams.mat'; %here is where we will save the learned parameters.

params = load('/Users/Davis/Desktop/garyComposite/params/preprocessparams.mat');
trainData = params.trainPCA;
trainLabels = params.trainLabels;
validData = params.validPCA;
validLabels = params.validLabels;
testData = params.testPCA;
testLabels = params.testLabels;


fprintf('setting up network... \n');

inlen = size(trainData,1);
numTrainImages = size(trainData,2);
numTestImages = size(testData, 2);
numValidImages = size(validData, 2);

maxim = 0;
for i = 1:size(testLabels,2)
    curr = testLabels{i};
    if curr>maxim
        maxim = curr;
    end 
end

targlen = maxim;
               
weights = (normrnd(0,1/sqrt(inlen+1),[targlen,inlen+1]));  %randomized weight matrix for hidden layer to output layer
learn = 0.0001;

a = .9;         %momentum constant

dwold = 0;     %old weight change matrix for hidden to output units

run = true;
epoch = 0;

show = 100;
%% Stochastic Gradient Descent Code

Error = [];

ValidPercentWrong = [];

TestPercentWrong = [];

%mainprogram

%% training
fprintf('training... \n');

while run
    
    epoch = epoch + 1;
    
    trainError = 0;
    validWrong = 0;
    testWrong = 0;

    if(mod(epoch,show) ==0)
        fprintf('epoch %i... \n', epoch);
    end
    
    %% do SGD for training images
    for i = randperm(numTrainImages)
        input = [1; trainData(:,i)];
        
        targ = zeros(targlen,1);
        targ(trainLabels{i}) = 1; %set up target

        %forward propagate function

        net = weights*input;
        
        %out = [1./(1+exp(-net))];
        %oprime = out.*(1-out);
        
        out = exp(net)./sum(exp(net));
        oprime = 1;
  
        deltao = (targ - out).*oprime;

        
        dw = (deltao*input') + dwold*a;
        weights = weights + learn.*(dw);
        
        

        dwold = dw;
        
        trainError = trainError + 0.5*sum((targ-out).^2)/numTrainImages; 
        
    end
    %% print training error
    Error = [Error, trainError];
    if (mod(epoch,show) == 0)
        fprintf('    training error: %f \n', Error(epoch));
    end
    
    
    %% forward propagate and find out validation error

    for i = randperm(numValidImages)        
        wrong = feedforwards(weights, validData(:,i), validLabels{i}, 0);
        validWrong = validWrong + wrong;
    end  
    
    %% print validation percent wrong
    
    vpWrong = 100*(validWrong/numValidImages);
    ValidPercentWrong = [ValidPercentWrong, vpWrong];
    if(mod(epoch,show) == 0)
        fprintf('    validation percent wrong: %%%f \n', ValidPercentWrong(epoch));
    end
    
   
 %% forwad propagate and find out test error

    for i = randperm(numTestImages)
        wrong = feedforwards(weights, testData(:,i), testLabels{i}, 0);
        testWrong = testWrong + wrong;
    end
    
    %% print test percent wrong
    
    tpWrong = 100*(testWrong/numTestImages);
    TestPercentWrong = [TestPercentWrong, tpWrong];
    if (mod(epoch,show) == 0)
        fprintf('    test percent wrong: %%%f \n', TestPercentWrong(epoch));
    end
    
    %% break when...
    if(epoch == numIter)
        run = false;
        fprintf('****************TRAINING ENDS*********************')
        plot(Error, 'b'); hold;
        plot(ValidPercentWrong, 'r');
        plot(TestPercentWrong, 'g');
        legend('Training: SSE', 'Validation: Percent Wrong', 'Test: Percent Wrong');
    end

end






    