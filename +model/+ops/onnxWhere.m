function [output, numDimsOutput] = onnxWhere(condition, X, Y, numDimsCondition, numDimsX, numDimsY)

% Copyright 2020 The MathWorks, Inc.

bigz = zeros(size(condition + X + Y));      % broadcast
condition = condition + bigz;
X = X + bigz;
output = Y + bigz;
output(condition==1) = X(condition==1);
numDimsOutput = max([numDimsCondition, numDimsX, numDimsY]);
end
