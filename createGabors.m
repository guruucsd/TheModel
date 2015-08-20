function [output] = createGabors(numScales, numOrientations)
% creates gabors of 5 sizes and 8 orientations for images that are 
% 256 by 256. Do the downsampling first, before filtering. and dwsp after.
% @author Davis Liang
% @version 1.0
% 7/13/15

display('building gabors filters');

%% gabor constants
frequency = 1.7;
gaborSize = 64; %the width and height of each gabor feature is 48.
std_dev = pi;
%% initializing gabor variables
scale = zeros(1,numScales);
orientation = zeros(1,numOrientations);

for i=1:numScales
    scale(i) = (2*pi/gaborSize)*1.6^(i);
end

for i=1:numOrientations
    orientation(i) = (pi/numOrientations)*(i-1);
end

carrier = zeros(gaborSize, gaborSize);
envelop = zeros(gaborSize, gaborSize);
gabor = zeros(gaborSize, gaborSize, numOrientations, numScales);

%% initializing
%note: 1i is one*i to differentiate the variable i from the imaginary num..
% scale is j, orientation is k, ii is y, j is x, phi is
% orientation, k is scale 

for j = 1:numScales
    for k = 1:numOrientations
        for y = -gaborSize+1:gaborSize
            for x = -gaborSize+1:gaborSize
                carrier(y+gaborSize, x+gaborSize) = exp(0.6*1i*(scale(j)*cos(orientation(k))*y+scale(j)*sin(orientation(k))*x));
                envelop(y+gaborSize, x+gaborSize) = exp(-(scale(j)^2*(y^2+x^2))/(2*std_dev*std_dev*frequency));
                gabor(y+gaborSize, x+gaborSize, k, j) = carrier(y+gaborSize, x+gaborSize)*envelop(y+gaborSize, x+gaborSize);
            end
        end
    end
end

%% organize into useable data cell
for s = 1:numScales
    for o = 1:numOrientations
        output{s,o} = gabor(:,:,o, s);
    end
end
% output{s,o} refers to gabor of scale s and orientation o.

%% done.
display('    gabors successfully built');

