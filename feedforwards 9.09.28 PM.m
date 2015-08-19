function [wrong, out] = feedforwards(whi,woh,input, label)

wrong = 0;

input = [1; input];


%forward propagate function
neti = [whi*input];
hout = [1./(1+exp(-neti))];
    
h_layer = [1; hout];
neto = woh*[h_layer];
    
out = 1./(1+exp(-neto));

[m, index_out] = max(out);
index_targ = label;

if index_targ ~= index_out
    wrong = 1;
end