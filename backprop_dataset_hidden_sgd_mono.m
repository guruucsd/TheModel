function [errors total_gradients hidden_RH]=backprop_dataset_hidden_sgd_mono(theta,nodes,input_size,num_output,num_train,num_hidden)
[W_in_hd1 W_hd_op1]=GetParameterMonoNetwork(theta,input_size,num_output,num_train,num_hidden);

%forward
networknode{1}=nodes;
networknode{1}.val=nodes.val;
networknode{1}.id=1;
%hidden layer
networknode{2}.id=2;
hidden_RH=W_in_hd1*networknode{1}.val;
networknode{2}.val=sigmoid(W_in_hd1*networknode{1}.val);    
networknode{2}.type='hidden';
%output layer
networknode{3}.id=3;
networknode{3}.val=W_hd_op1*networknode{2}.val;
networknode{3}.type='output';

errors=sum((networknode{1}.dsoutput-networknode{3}.val).^2);

%backpropagating...
 for nodeid=3:-1:1
     currentNode = networknode{nodeid};
     if strcmp(currentNode.type,'output')
        currentNode.delta=-2*(networknode{1}.dsoutput-currentNode.val);
        grad_W_hd_op1=currentNode.delta*networknode{2}.val';
        networknode{nodeid}.delta=currentNode.delta;
     elseif strcmp(currentNode.type,'hidden')
        gr_sigmoid=networknode{2}.val.*(1-networknode{2}.val);
        currentNode.delta=((W_hd_op1'*networknode{3}.delta).*gr_sigmoid);
        grad_W_in_hd1=currentNode.delta*networknode{1}.val';               
        networknode{nodeid}.delta=currentNode.delta;
     end
 end
 total_gradients = [grad_W_in_hd1(:); grad_W_hd_op1(:);];


