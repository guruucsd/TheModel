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
               
weights = (normrnd(0,1/sqrt(inlen),[targlen,inlen]));   %randomized weight matrix for hidden layer to output layer
bias = ones(targlen,1)*0.5;                             %bias weight matrix
learn = 0.0001;

a = .9;         %momentum constant
lambda = 0.3;   %weight decay constant

d_weights_old = 0;     %old weight change matrix for hidden to output units
d_bias_old = 0;         

run = true;
epoch = 0;

show = 1000;
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
    
    affectWrong = zeros(2,7);

    if(mod(epoch,show) ==0)
        fprintf('epoch %i... \n', epoch);
    end
    
    %% do SGD for training images
    for i = randperm(numTrainImages)
        input = [trainData(:,i)];
        
        targ = zeros(targlen,1);
        targ(trainLabels{i}) = 1; %set up target

        %forward propagate function

        net = weights*input + bias;
        
        %out = 1./(1+exp(-net));
        %oprime = out.*(1-out);
        
        out = exp(net)./sum(exp(net));
        oprime = 1;
  
        deltao = (targ - out).*oprime;

        regularizer = (lambda/numTrainImages)*weights;
        
        d_weights = (deltao*input');
        d_bias = (deltao*1);
        weights = weights + learn.*(d_weights - regularizer + d_weights_old*a);
        bias = bias + learn.*(d_bias + d_bias_old*a);
        
        d_bias_old = d_bias;
        d_weights_old = d_weights;
        
        
        
        %sse = sum((targ-out).^2);
        CrossEntropyCost = (-1/numTrainImages)*sum(targ.*log(out)+(1-targ).*log(1-out));
        
        trainError = trainError + CrossEntropyCost; 
        
    end
    %% print training error
    Error = [Error, trainError];
    if (mod(epoch,show) == 0)
        fprintf('    training error: %f \n', Error(epoch));
    end
    
    
    %% forward propagate and find out validation error

    for i = randperm(numValidImages)        
        wrong = feedforwards(weights, bias, validData(:,i), validLabels{i}, 0);
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
        wrong = feedforwards(weights, bias, testData(:,i), testLabels{i}, 0);
        testWrong = testWrong + wrong;
        if wrong == 1
            if i <= size(testData,2)/2
                affectWrong(1,testLabels{i}) = affectWrong(1,testLabels{i}) + 1; % top
            else
                affectWrong(2,testLabels{i}) = affectWrong(2,testLabels{i}) + 1;
            end
        end
    end
    
    %% print test percent wrong
    
    tpWrong = 100*(testWrong/numTestImages);
    TestPercentWrong = [TestPercentWrong, tpWrong];
    if (mod(epoch,show) == 0)
        fprintf('    test percent wrong: %%%f \n', TestPercentWrong(epoch));
    end
    
    %% print test percent wrong as per each facial affect and top/bottom category
    
    %1= afraid, 2=angry, 3=disgusted, 4=happy, 5=neutral , 6=sad,
    %7=surprised
    if (mod(epoch,show)==0)
       fprintf('    AFRAID_TOP_ERROR: %%%f \n', affectWrong(1,1)/4);
       fprintf('    AFRAID_BOT_ERROR: %%%f \n', affectWrong(2,1)/4);
       
       fprintf('    ANGRY_TOP_ERROR: %%%f \n', affectWrong(1,2)/4);
       fprintf('    ANGRY_BOT_ERROR: %%%f \n', affectWrong(2,2)/4);
       
       fprintf('    DISGUSTED_TOP_ERROR: %%%f \n', affectWrong(1,3)/4);
       fprintf('    DISGUSTED_BOT_ERROR: %%%f \n', affectWrong(2,3)/4);
       
       fprintf('    HAPPY_TOP_ERROR: %%%f \n', affectWrong(1,4)/4);
       fprintf('    HAPPY_BOT_ERROR: %%%f \n', affectWrong(2,4)/4);
       
       fprintf('    SAD_TOP_ERROR: %%%f \n', affectWrong(1,6)/4);
       fprintf('    SAD_BOT_ERROR: %%%f \n', affectWrong(2,6)/4);
       
       fprintf('    SURPRISED_TOP_ERROR: %%%f \n', affectWrong(1,7)/4);
       fprintf('    SURPRISED_BOT_ERROR: %%%f \n', affectWrong(2,7)/4);
       
       
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






    