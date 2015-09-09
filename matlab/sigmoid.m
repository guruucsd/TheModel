function [value]=sigmoid(input)
value=1./(1+exp(-input));