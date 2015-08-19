function[] = CompositeClassifierTrain(numIter, numhid)



savePath = '/Users/Davis/Desktop/garyComposite/params/networkparams.mat'; %here is where we will save the learned parameters.

params = load('/Users/Davis/Desktop/garyComposite/params/preprocessparams.mat');
trainData = params.trainPCA;
trainLabels = params.trainLabels;
validData = params.validPCA;
validLabels = params.validLabels;
testData = params.testPCA;
testLabels = params.testLabels;


maxDat = max(max(trainData));

trainData = 2*(trainData/maxDat)-1;
testData = 2*(testData/maxDat)-1;
validData = 2*(validData/maxDat)-1;



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

whi = (normrnd(0,1/sqrt(inlen+1),[numhid,inlen+1]));    %randomized weight matrix for input layer to hidden layer                 
woh = (normrnd(0,1/sqrt(numhid+1),[targlen,numhid+1]));  %randomized weight matrix for hidden layer to output layer
learnho = 0.0000025;
learnih = 0.000005;
a = .9;         %momentum constant
dwoldh = 0;     %old weight change matrix for input to hidden units  
dwoldo = 0;     %old weight change matrix for hidden to output units

run = true;
epoch = 0;
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

    if(mod(epoch,100) ==0)
        fprintf('epoch %i... \n', epoch);
    end
    
    %% do SGD for training images
    for i = randperm(numTrainImages)
        input = [1; trainData(:,i)];
        
        targ = zeros(targlen,1);
        targ(trainLabels{i}) = 1; %set up target

        %forward propagate function
        neti = [whi*input];
        hout = [1./(1+exp(-neti))];
        hprime = hout.*(1-hout);
        
        h_layer = [1; hout];
        neto = woh*[h_layer];
        
        %out = [1./(1+exp(-neto))];
        %oprime = out.*(1-out);
        
        out = exp(neto)./sum(exp(neto));
        oprime = 1;
        
        
        
        deltao = (targ - out).*oprime;
        deltah = hprime.*(woh(:,2:numhid+1)'*deltao);
        
        dwo = (deltao*[h_layer]') + dwoldo*a;
        woh = woh + learnho.*(dwo );
        
        dwh = ((deltah)*input') + dwoldh*a;
        whi = whi + learnih.*(dwh );
        
        trainError = trainError + 0.5*sum((targ-out).^2)/numTrainImages; 

        
    end
    %% print training error
    Error = [Error, trainError];
    if (mod(epoch,100) == 0)
        fprintf('    training error: %f \n', Error(epoch));
    end
    
    
    %% forward propagate and find out validation error

    for i = randperm(numValidImages)
        wrong = feedforwards(whi, woh, validData(:,i), validLabels{i});
        validWrong = validWrong + wrong;
    end  
    
    %% print validation percent wrong
    
    vpWrong = 100*(validWrong/numValidImages);
    ValidPercentWrong = [ValidPercentWrong, vpWrong];
    if(mod(epoch,100) == 0)
        fprintf('    validation percent wrong: %%%f \n', ValidPercentWrong(epoch));
    end
    
   
 %% forwad propagate and find out test error

    for i = randperm(numTestImages)
        wrong = feedforwards(whi, woh, testData(:,i), testLabels{i});
        testWrong = testWrong + wrong;
    end
    
    %% print test percent wrong
    
    tpWrong = 100*(testWrong/numTestImages);
    TestPercentWrong = [TestPercentWrong, tpWrong];
    if (mod(epoch,100) == 0)
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






    