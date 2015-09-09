function [wrong, out] = feedforwards(weight, bias, input, label, show)

wrong = 0;


neto = weight*input+bias;
    
out = 1./(1+exp(-neto));

[m, index_out] = max(out);
index_targ = label;

if(show==1)
    affectVector = {'afraid', 'angry', 'disgusted', 'happy', 'neutral', 'sad', 'suprised'};
    fprintf('the correct label is: %s (%i) \n', affectVector{label},label);
    fprintf('your chosen label is: %s (%i) \n', affectVector{index_out},index_out);
    out
end



if index_targ ~= index_out
    wrong = 1;
end

