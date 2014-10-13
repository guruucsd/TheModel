function [W_in_hd1 W_hd_op1]=GetParameterMonoNetwork(theta,input_size,num_output,num_train,num_hidden)
    W_in_hd1=reshape(theta(1:num_hidden*input_size),num_hidden,input_size);
    in_hd=num_hidden*input_size;
    W_hd_op1=reshape(theta(in_hd+1:in_hd+num_output*num_hidden),num_output,num_hidden);
end
    
    
