function [Y, numDimsY] = onnxNonZero(X, numDimsX)

% Copyright 2020 The MathWorks, Inc.

% Implements the ONNX NonZero operator
Coords      = cell(1,max(2,numDimsX));
% Extract the coordinates of the nonzero points 
[Coords{:}] = ind2sub(size(X, 1:max(2,numDimsX)), find(extractdata(X)~=0));
% Concatenate the coordinates into a 2D matrix
Y = [Coords{:}];
% Subtract 1 to convert to origin 0
Y = Y - 1;
if numDimsX < 2
    Y = Y(:,1);
end
Y = dlarray(Y);
numDimsY = 2;
end
